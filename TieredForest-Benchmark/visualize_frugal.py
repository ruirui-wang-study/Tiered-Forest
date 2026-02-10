"""
可视化 FrugalGPT 工作流程和实验结果
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_frugal_workflow():
    """绘制 FrugalGPT 工作流程图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(5, 9.5, 'FrugalGPT 工作流程', 
            ha='center', va='top', fontsize=20, fontweight='bold')
    
    # 1. 用户问题
    box1 = FancyBboxPatch((3.5, 8), 3, 0.6, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(box1)
    ax.text(5, 8.3, '用户问题', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 箭头 1
    arrow1 = FancyArrowPatch((5, 8), (5, 7.2), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # 2. 小模型 (Tier 1)
    box2 = FancyBboxPatch((3, 6.2), 4, 0.8, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box2)
    ax.text(5, 6.8, '小模型 (Qwen2.5-7B)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.4, '成本: $0.0002/1K tokens', ha='center', va='center', fontsize=9)
    
    # 箭头 2
    arrow2 = FancyArrowPatch((5, 6.2), (5, 5.4), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # 3. 评分函数
    box3 = FancyBboxPatch((3.5, 4.6), 3, 0.6, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(box3)
    ax.text(5, 4.9, '评分函数: Score = 0.8', ha='center', va='center', fontsize=11)
    
    # 箭头 3a (质量够好)
    arrow3a = FancyArrowPatch((6.5, 4.9), (8, 4.9), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow3a)
    ax.text(7.2, 5.2, 'Score ≥ 0.7', ha='center', va='center', fontsize=9, color='green')
    
    # 返回答案 (右侧)
    box_return = FancyBboxPatch((8, 4.4), 1.5, 1, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box_return)
    ax.text(8.75, 4.9, '返回\n答案', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 箭头 3b (质量不够)
    arrow3b = FancyArrowPatch((5, 4.6), (5, 3.8), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow3b)
    ax.text(5.5, 4.2, 'Score < 0.7', ha='left', va='center', fontsize=9, color='red')
    
    # 4. 大模型 (Tier 2)
    box4 = FancyBboxPatch((3, 2.8), 4, 0.8, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 3.4, '大模型 (DeepSeek-Chat)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 3.0, '成本: $0.002/$0.008 per 1K tokens', ha='center', va='center', fontsize=9)
    
    # 箭头 4
    arrow4 = FancyArrowPatch((5, 2.8), (5, 2.0), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # 5. 最终答案
    box5 = FancyBboxPatch((3.5, 1.2), 3, 0.6, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(box5)
    ax.text(5, 1.5, '返回最终答案', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='green', label='便宜模型'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='red', label='昂贵模型'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='orange', label='评分函数'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    # 添加说明
    ax.text(0.5, 0.5, '核心思想: 简单问题用便宜模型，复杂问题用贵模型', 
            ha='left', va='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('results/frugal_workflow.png', dpi=300, bbox_inches='tight')
    print("✓ 工作流程图已保存: results/frugal_workflow.png")
    plt.close()


def plot_cost_comparison():
    """绘制成本对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 数据
    methods = ['Naive LLM\n(只用大模型)', 'FrugalGPT\n(简化版)', 'Tiered-Forest\n(符号层)']
    costs = [0.0005, 0.0002, 0.0001]  # 平均成本 (USD)
    accuracies = [0.95, 0.94, 0.96]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    # 绘制柱状图
    x = np.arange(len(methods))
    bars = ax.bar(x, costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 添加准确率标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.00002,
                f'准确率: {acc:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 添加成本标签
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'${height:.4f}',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax.set_ylabel('平均成本 (USD/问题)', fontsize=12, fontweight='bold')
    ax.set_title('FrugalGPT vs Baselines - 成本对比', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加成本节省标注
    ax.annotate('', xy=(0, costs[0]), xytext=(1, costs[1]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, (costs[0] + costs[1])/2 + 0.00003, 
            f'节省 {(1-costs[1]/costs[0])*100:.0f}%', 
            ha='center', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/frugal_cost_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 成本对比图已保存: results/frugal_cost_comparison.png")
    plt.close()


def plot_cascade_distribution():
    """绘制级联分布图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 饼图 - 停止位置分布
    labels = ['小模型停止', '大模型停止']
    sizes = [33.3, 66.7]
    colors = ['#4ecdc4', '#ff6b6b']
    explode = (0.1, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('FrugalGPT 停止位置分布\n(测试集)', fontsize=14, fontweight='bold', pad=20)
    
    # 右图: 柱状图 - 平均 LLM 调用次数
    methods = ['Naive LLM', 'FrugalGPT']
    llm_calls = [1.0, 1.67]
    colors_bar = ['#ff6b6b', '#4ecdc4']
    
    bars = ax2.bar(methods, llm_calls, color=colors_bar, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    for bar, calls in zip(bars, llm_calls):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{calls:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('平均 LLM 调用次数', fontsize=12, fontweight='bold')
    ax2.set_title('平均 LLM 调用次数对比', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, 2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/frugal_cascade_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ 级联分布图已保存: results/frugal_cascade_distribution.png")
    plt.close()


def main():
    """生成所有可视化"""
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*60)
    print("生成 FrugalGPT 可视化图表")
    print("="*60 + "\n")
    
    plot_frugal_workflow()
    plot_cost_comparison()
    plot_cascade_distribution()
    
    print("\n" + "="*60)
    print("所有图表生成完成！")
    print("="*60)
    print("\n查看图表:")
    print("  - results/frugal_workflow.png")
    print("  - results/frugal_cost_comparison.png")
    print("  - results/frugal_cascade_distribution.png")
    print()


if __name__ == "__main__":
    main()
