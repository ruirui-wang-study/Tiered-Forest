"""
训练 FrugalGPT 评分模型

使用 DistilBERT 训练一个回归模型来评估 LLM 答案的质量

输入: (question, answer)
输出: 质量分数 [0, 1]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
from typing import List, Dict


class ScorerDataset(Dataset):
    """评分模型数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 拼接 question 和 answer
        text = f"{item['question']} [SEP] {item['answer']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.float)
        }


def load_data(data_file: str) -> List[Dict]:
    """加载训练数据"""
    print(f"加载数据: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  总样本数: {len(data)}")
    print(f"  正确样本: {sum(1 for d in data if d['label'] == 1.0)}")
    print(f"  错误样本: {sum(1 for d in data if d['label'] == 0.0)}")
    
    return data


def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    """计算评估指标"""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # 将预测值限制在 [0, 1]
    predictions = np.clip(predictions, 0, 1)
    
    # 回归指标
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    
    # 分类指标 (阈值 0.5)
    pred_binary = (predictions >= 0.5).astype(int)
    labels_binary = (labels >= 0.5).astype(int)
    accuracy = accuracy_score(labels_binary, pred_binary)
    
    return {
        'mse': mse,
        'mae': mae,
        'accuracy': accuracy
    }


def train_scorer(
    data_file: str,
    output_dir: str = "models/frugal_scorer",
    test_size: float = 0.2,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """
    训练评分模型
    
    Args:
        data_file: 训练数据文件
        output_dir: 模型输出目录
        test_size: 测试集比例
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    print("=" * 60)
    print("训练 FrugalGPT 评分模型")
    print("=" * 60)
    
    # 1. 加载数据
    data = load_data(data_file)
    
    # 2. 划分训练集和验证集
    train_data, val_data = train_test_split(
        data, test_size=test_size, random_state=42, stratify=[d['label'] for d in data]
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    
    # 3. 加载 tokenizer 和模型
    print(f"\n加载模型...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=1  # 回归任务
    )
    
    # 4. 创建数据集
    train_dataset = ScorerDataset(train_data, tokenizer)
    val_dataset = ScorerDataset(val_data, tokenizer)
    
    # 5. 训练配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy='epoch',  # 修改为 eval_strategy
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='mse',
        greater_is_better=False,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        save_total_limit=2,
        report_to='none',  # 不使用 wandb 等
    )
    
    # 6. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 7. 训练
    print(f"\n开始训练...")
    print(f"  轮数: {num_epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    
    trainer.train()
    
    # 8. 评估
    print(f"\n评估模型...")
    eval_results = trainer.evaluate()
    
    print(f"\n评估结果:")
    print(f"  MSE: {eval_results['eval_mse']:.4f}")
    print(f"  MAE: {eval_results['eval_mae']:.4f}")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.2%}")
    
    # 9. 保存模型
    print(f"\n保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 10. 保存训练历史
    history_file = f"{output_dir}/training_history.json"
    with open(history_file, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    print(f"✓ 训练历史已保存: {history_file}")
    
    # 11. 绘制训练曲线
    plot_training_history(trainer.state.log_history, output_dir)
    
    return trainer, eval_results


def plot_training_history(history: List[Dict], output_dir: str):
    """绘制训练曲线"""
    train_loss = []
    eval_loss = []
    eval_accuracy = []
    
    for entry in history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
        if 'eval_accuracy' in entry:
            eval_accuracy.append(entry['eval_accuracy'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲线
    ax1.plot(train_loss, label='Train Loss', marker='o')
    if eval_loss:
        epochs = np.linspace(0, len(train_loss), len(eval_loss))
        ax1.plot(epochs, eval_loss, label='Eval Loss', marker='s')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy 曲线
    if eval_accuracy:
        ax2.plot(eval_accuracy, label='Eval Accuracy', marker='s', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = f"{output_dir}/training_curves.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {plot_file}")
    plt.close()


def test_model(model_dir: str, test_questions: List[tuple]):
    """测试训练好的模型"""
    print("\n" + "=" * 60)
    print("测试训练好的模型")
    print("=" * 60)
    
    # 加载模型
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    print(f"\n测试样例:")
    for question, answer in test_questions:
        text = f"{question} [SEP] {answer}"
        
        inputs = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Score: {score:.3f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练 FrugalGPT 评分模型")
    parser.add_argument("--data", type=str, required=True,
                       help="训练数据文件路径")
    parser.add_argument("--output", type=str, default="models/frugal_scorer",
                       help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="测试集比例")
    
    args = parser.parse_args()
    
    try:
        # 训练模型
        trainer, eval_results = train_scorer(
            data_file=args.data,
            output_dir=args.output,
            test_size=args.test_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        
        # 测试模型
        test_questions = [
            ("Who directed Inception?", "Christopher Nolan directed Inception."),
            ("Who directed Inception?", "I don't know."),
            ("What is 2+2?", "4"),
            ("What is 2+2?", "The answer is unclear."),
        ]
        
        test_model(args.output, test_questions)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"\n模型已保存到: {args.output}")
        print(f"\n下一步: 使用训练好的模型")
        print(f"  python test_trained_scorer.py --model {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
