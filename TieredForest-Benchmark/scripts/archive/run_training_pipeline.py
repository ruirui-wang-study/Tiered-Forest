"""
完整的评分模型训练流程

步骤:
1. 收集训练数据
2. 训练 DistilBERT 模型
3. 测试模型
4. 集成到 FrugalGPT

一键运行所有步骤
"""

import subprocess
import sys
import os
import json


def run_command(cmd: str, description: str):
    """运行命令并显示进度"""
    print("\n" + "=" * 60)
    print(f"步骤: {description}")
    print("=" * 60)
    print(f"命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ 失败: {description}")
        sys.exit(1)
    
    print(f"\n✓ 完成: {description}")


def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    
    required = [
        'torch',
        'transformers',
        'sklearn',
        'tqdm',
        'matplotlib'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print(f"\n安装命令:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("✓ 所有依赖已安装")
    return True


def main():
    """主流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description="完整的评分模型训练流程")
    parser.add_argument("--max-samples", type=int, default=200,
                       help="收集的最大样本数 (默认: 200)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="训练轮数 (默认: 3)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="批次大小 (默认: 16)")
    parser.add_argument("--skip-collect", action="store_true",
                       help="跳过数据收集步骤（如果已有数据）")
    parser.add_argument("--skip-train", action="store_true",
                       help="跳过训练步骤（如果已有模型）")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("FrugalGPT 评分模型训练流程")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装依赖:")
        print("  pip install torch transformers scikit-learn tqdm matplotlib")
        return
    
    # 定义路径
    data_file = "data/training/scorer_training_data.json"
    model_dir = "models/frugal_scorer"
    
    # 步骤 1: 收集训练数据
    if not args.skip_collect:
        if os.path.exists(data_file):
            response = input(f"\n训练数据已存在: {data_file}\n是否重新收集? (y/N): ")
            if response.lower() != 'y':
                print("跳过数据收集")
                args.skip_collect = True
        
        if not args.skip_collect:
            run_command(
                f"python collect_training_data.py --max-samples {args.max_samples} --output {data_file}",
                "收集训练数据"
            )
    else:
        print("\n跳过数据收集步骤")
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"\n❌ 训练数据不存在: {data_file}")
        print("请先运行数据收集步骤")
        return
    
    # 显示数据统计
    with open(data_file, 'r') as f:
        data = json.load(f)
    print(f"\n训练数据统计:")
    print(f"  总样本数: {len(data)}")
    print(f"  正确样本: {sum(1 for d in data if d['label'] == 1.0)}")
    print(f"  错误样本: {sum(1 for d in data if d['label'] == 0.0)}")
    
    # 步骤 2: 训练模型
    if not args.skip_train:
        if os.path.exists(model_dir):
            response = input(f"\n模型已存在: {model_dir}\n是否重新训练? (y/N): ")
            if response.lower() != 'y':
                print("跳过模型训练")
                args.skip_train = True
        
        if not args.skip_train:
            run_command(
                f"python train_scorer.py --data {data_file} --output {model_dir} "
                f"--epochs {args.epochs} --batch-size {args.batch_size}",
                "训练评分模型"
            )
    else:
        print("\n跳过模型训练步骤")
    
    # 检查模型
    if not os.path.exists(model_dir):
        print(f"\n❌ 模型不存在: {model_dir}")
        print("请先运行训练步骤")
        return
    
    # 步骤 3: 测试模型
    print("\n" + "=" * 60)
    print("测试训练好的模型")
    print("=" * 60)
    
    from src.agents.trained_scorer import TrainedScorer
    
    scorer = TrainedScorer(model_dir)
    
    test_cases = [
        ("Who directed Inception?", "Christopher Nolan directed Inception."),
        ("Who directed Inception?", "I don't know."),
        ("What is 2+2?", "4"),
        ("What is 2+2?", "Maybe 5?"),
    ]
    
    print("\n测试案例:")
    for question, answer in test_cases:
        score = scorer.score(question, answer)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Score: {score:.3f}")
    
    # 完成
    print("\n" + "=" * 60)
    print("训练流程完成！")
    print("=" * 60)
    
    print(f"\n模型已保存到: {model_dir}")
    print(f"\n使用方法:")
    print(f"  from src.agents.trained_scorer import TrainedScorer")
    print(f"  scorer = TrainedScorer('{model_dir}')")
    print(f"  score = scorer.score(question, answer)")
    
    print(f"\n集成到 FrugalGPT:")
    print(f"  修改 frugal_agent.py，将 SimpleScorer 替换为 TrainedScorer")
    
    print(f"\n下一步:")
    print(f"  1. 运行 test_frugal.py 测试集成")
    print(f"  2. 运行 run_benchmark.py 进行完整评估")


if __name__ == "__main__":
    main()
