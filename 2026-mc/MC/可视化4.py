import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from scipy.stats import pearsonr

# ============================================
# 1. 基础样式加载 (参考 6.py)
# ============================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#ffffff00"

# 加载数据
file_path = "data/测试集预测结果.xlsx" # 请确保路径正确
df = pd.read_excel(file_path)

# 选择要分析的核心变量
cols = ['HDL', 'TC', 'LDL', '血尿酸', 'TG', '血瘀质', '真实标签']
data_corr = df[cols].corr() # 计算相关系数矩阵

# ============================================
# 2. 辅助函数 (参考 6.py)
# ============================================
def get_mantel_style(r_val, p_val):
    """根据相关性和显著性定义连线样式"""
    # 定义粗细 (Mantel r)
    if r_val >= 0.4: width = 4
    elif r_val >= 0.2: width = 2
    else: width = 1
    
    # 定义颜色 (p值)
    if p_val < 0.01: color = "#f7a08b" # 极显著
    elif p_val < 0.05: color = "#90be6d" # 显著
    else: color = "#a9a9a9" # 不显著
    return width, color

# ============================================
# 3. 绘图核心逻辑
# ============================================
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
ax.set_aspect('equal')

n = len(cols)
# 绘制底层的相关性方块矩阵 (下三角)
pearson_colors = ["#5b85ef", "#76a6ba", "#bcfbee"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", pearson_colors)

for i in range(n):
    for j in range(i + 1):
        r = data_corr.iloc[i, j]
        # 绘制背景方块
        rect = patches.Rectangle((j, n - i - 1), 1, 1, facecolor=cmap((r + 1) / 2), 
                                 edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        # 标注数值
        ax.text(j + 0.5, n - i - 0.5, f"{r:.2f}", ha='center', va='center', 
                color="white" if abs(r) > 0.5 else "black", fontsize=9)

# 设置轴标签
ax.set_xticks(np.arange(n) + 0.5)
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticks(np.arange(n) + 0.5)
ax.set_yticklabels(cols[::-1])

# ============================================
# 4. 模拟绘制网络连线 (示范核心变量：真实标签)
# ============================================
# 假设我们要研究 "真实标签" 与其他指标的 Mantel-like 关联
target_idx = n - 1 # '真实标签' 的索引
for i in range(n - 1):
    r_val, p_val = pearsonr(df[cols[target_idx]], df[cols[i]])
    lw, col = get_mantel_style(abs(r_val), p_val)
    
    # 绘制从侧边出发的曲线
    start_pos = (n - 0.5, n - target_idx - 0.5)
    end_pos = (i + 0.5, n - i)
    
    # 使用弧线模拟网络连接
    connection = patches.ConnectionPatch(
        xyA=(n, n - target_idx - 0.5), xyB=(i + 0.5, n - i),
        coordsA="data", coordsB="data",
        axesA=ax, axesB=ax,
        connectionstyle="arc3,rad=-0.3",
        color=col, linewidth=lw, alpha=0.7, zorder=1
    )
    ax.add_artist(connection)

# ============================================
# 5. 图例系统 (参考 6.py)
# ============================================
# 相关性颜色条
norm = mcolors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=10, pad=0.1)
cbar.set_label("Pearson's r", fontsize=10)

# 添加自定义连线图例
p_handles = [
    mlines.Line2D([], [], color="#faf2ba", label="< 0.01"),
    mlines.Line2D([], [], color="#beb56d", label="0.01 - 0.05"),
    mlines.Line2D([], [], color="#a9a9a9", label=">= 0.05"),
]
ax.legend(handles=p_handles, title="P-value (Mantel's p)", 
          bbox_to_anchor=(1.3, 1), loc='upper left', frameon=False)

ax.set_title("指标间相关性网络热图 (Correlation Network)", fontsize=15, pad=30, fontweight='bold')
ax.set_xlim(0, n + 2) # 留出连线空间
plt.tight_layout()
plt.show()