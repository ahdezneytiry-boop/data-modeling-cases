import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================
# 1. 基础配置：设置中文字体与全局样式
# ============================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.facecolor'] = '#f8f9fa'  # 轻微灰色背景，更具质感

# 颜色方案：珊瑚红（高风险）、青色（低风险）、灰蓝（中风险）
colors_risk = ['#e16259', '#9dc1c5', '#8e93af'] 
blue_main = "#084594" # 深蓝色用于箱线图

# ============================================
# 2. 数据准备 (基于你提供的 xlsx 数据)
# ============================================
# 环形图数据
risk_labels = ['高风险', '低风险', '中风险']
risk_sizes = [160, 38, 2]

# 箱线图统计字典 (必须包含 fliers 键以修正 KeyError)
prob_stats = [{
    'label': '预测概率分布',
    'whislo': 0.005,    # 最小值
    'q1': 0.96875,      # 25%分位数
    'med': 0.995,       # 中位数
    'q3': 1.0,          # 75%分位数
    'whishi': 1.0,      # 最大值
    'fliers': []        # 必须为空列表，修正报错
}]

# ============================================
# 3. 开始绘图
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

# --- A. 左图：风险等级分布环形图 ---
wedges, texts, autotexts = ax1.pie(
    risk_sizes, labels=risk_labels, autopct='%1.1f%%',
    startangle=140, colors=colors_risk,
    pctdistance=0.85, explode=(0.05, 0, 0.2), # 突出高风险和中风险
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)

# 绘制中心白圆，形成环形感
centre_circle = plt.Circle((0,0), 0.70, fc='white')
ax1.add_artist(centre_circle)
ax1.set_title("样本风险等级占比分布 (N=200)", fontsize=15, pad=20, fontweight='bold')

# --- B. 右图：预测概率统计特征箱线图 ---
# 使用 bxp 绘制，修正了颜色覆盖警告
ax2.bxp(prob_stats, vert=False, patch_artist=True,
        boxprops=dict(facecolor='#6baed6', edgecolor=blue_main, alpha=0.8),
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(color=blue_main, linestyle='--', linewidth=1.2),
        capprops=dict(color=blue_main, linewidth=1.2))

# 动态标注关键数值
ax2.text(0.995, 1.15, f"中位数: 0.995", ha='center', color=blue_main, fontweight='bold', fontsize=10)
ax2.text(0.005, 0.82, f"Min: 0.005", ha='left', fontsize=9, color='#555555')
ax2.text(1.0, 0.82, f"Max: 1.0", ha='right', fontsize=9, color='#555555')

ax2.set_title("模型预测概率统计特征", fontsize=15, pad=20, fontweight='bold')
ax2.set_xlabel("模型预测患病概率 (P)", fontsize=12)
ax2.set_xlim(-0.05, 1.1)
ax2.grid(axis='x', linestyle='--', alpha=0.3)

# ============================================
# 4. 细节美化与输出
# ============================================
# 移除坐标轴边框
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# 保存高质量图片 (可选)
# plt.savefig("最终可视化分析图.png", dpi=300, bbox_inches='tight')

plt.show()