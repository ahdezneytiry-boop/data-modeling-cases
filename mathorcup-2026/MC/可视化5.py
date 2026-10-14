import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ============================================
# 1. 基础配置与数据加载
# ============================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 载入你的测试集预测结果
file_path = r"C:\Users\ahdezneytiry-boop1\AppData\Local\Programs\Microsoft VS Code\问题二_最终结果\测试集预测结果.xlsx"
df = pd.read_excel(file_path)
y_prob = df['预测概率']

# 你的实验设定阈值
T1 = 0.8240217278412527
T2 = 0.9994808980320797

# ============================================
# 2. 绘图实施
# ============================================
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

# 绘制概率分布密度图 (KDE) 和 直方图
sns.histplot(y_prob, bins=5, kde=True, color="#e0e5ff", alpha=0.4, ax=ax, label='样本分布')

# 填充颜色区间来代表风险等级 (使用科研配色)
ax.axvspan(0, T1, color="#a6bcff", alpha=0.2, label='低风险区')   # 青色
ax.axvspan(T1, T2, color="#6056d4", alpha=0.2, label='中风险区') # 黄色
ax.axvspan(T2, 1.0, color="#2b379f", alpha=0.2, label='高风险区') # 粉色

# 绘制阈值垂直分割线
ax.axvline(T1, color='#4d908e', linestyle='--', lw=2, label=f'T1 (低→中): {T1:.3f}')
ax.axvline(T2, color='#f94144', linestyle='--', lw=2, label=f'T2 (中→高): {T2:.3f}')

# 添加文字标注：各区间人数（基于你提供的数据）
# 低风险: 38, 中风险: 2, 高风险: 160
ax.text(T1/2, ax.get_ylim()[1]*0.8, '低风险\n(n=38)', ha='center', fontsize=12, fontweight='bold', color='#2f5e5c')
ax.text((T1+T2)/2, ax.get_ylim()[1]*0.8, '中风险\n(n=2)', ha='center', fontsize=12, fontweight='bold', color='#8a6d3b')
ax.text((T2+1.0)/2, ax.get_ylim()[1]*0.8, '高风险\n(n=160)', ha='center', fontsize=12, fontweight='bold', color='#a13d4a')

# ============================================
# 3. 样式美化
# ============================================
ax.set_title("基于双阈值的预测风险分层分布图", fontsize=15, pad=20, fontweight='bold')
ax.set_xlabel("模型预测患病概率 ($P$)", fontsize=12)
ax.set_ylabel("样本频数", fontsize=12)
ax.set_xlim(0, 1.05)

# 隐藏顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 自定义图例
ax.legend(loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
# plt.savefig("风险分层可视化.png", dpi=300)
plt.show()