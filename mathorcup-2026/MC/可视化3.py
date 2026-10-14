import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator

# ============================================
# 1. 基础配置与数据载入 (严格锁定你的数据)
# ============================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 你的实验数据
models = ['LR', 'RF', 'LightGBM']
auc_means = [0.599117, 0.854415, 0.904953]
auc_std = [0.021, 0.018, 0.015]  # 你提供的方差/标准差
recall_values = [0.996838, 0.958980, 0.902200]

# ============================================
# 2. 渐变色生成逻辑 (参考 05.py)
# ============================================
def create_gradient_colors(hex_colors, data):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0

    colors_rgb = [hex_to_rgb(color) for color in hex_colors]
    num_colors = len(colors_rgb)
    min_val, max_val = min(data), max(data)
    
    # 防止除以零
    if max_val == min_val: return [colors_rgb[0]] * len(data)

    def interpolate_color(val):
        if val <= min_val: return colors_rgb[0]
        if val >= max_val: return colors_rgb[-1]
        
        rel_pos = (val - min_val) / (max_val - min_val) * (num_colors - 1)
        idx = int(rel_pos)
        frac = rel_pos - idx
        return (1 - frac) * colors_rgb[idx] + frac * colors_rgb[idx + 1]

    return [interpolate_color(val) for val in data]

# 定义科研配色方案 (雅致渐变)
hex_colors = ["#345234", "#638856", "#75d267", "#60a193"] 
colors = create_gradient_colors(hex_colors, auc_means)

# ============================================
# 3. 绘图实施
# ============================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# 绘制柱状图 (带误差条)
bars = ax.bar(models, auc_means, yerr=auc_std, color=colors, 
              capsize=8, error_kw={'elinewidth':2, 'ecolor':'#444444'}, 
              edgecolor='white', linewidth=1.5, zorder=3)

# 添加数值标签 (AUC & Recall)
for i, bar in enumerate(bars):
    height = bar.get_height()
    # 标注 AUC
    ax.text(bar.get_x() + bar.get_width()/2., height + auc_std[i] + 0.01,
            f'AUC: {auc_means[i]:.3f}', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')
    # 标注 Recall (用不同颜色区分)
    ax.text(bar.get_x() + bar.get_width()/2., height / 2,
            f'Recall\n{recall_values[i]:.3f}', ha='center', va='center', 
            color='white', fontsize=10, fontweight='semibold')

# ============================================
# 4. 样式美化 (参考 05.py)
# ============================================
# 设置网格
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# 隐藏部分边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

# 添加 Colorbar
cmap = LinearSegmentedColormap.from_list("custom_cmap", hex_colors)
norm = Normalize(vmin=min(auc_means), vmax=max(auc_means))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.05)
cbar.set_label('Model Performance Index (AUC)', fontsize=10)
cbar.outline.set_visible(False)

# 标题与标签
ax.set_title("不同预测模型性能指标对比 (AUC & Recall)", fontsize=15, pad=20, fontweight='bold')
ax.set_ylabel("指标得分", fontsize=12, labelpad=10)
ax.set_ylim(0, 1.1)

plt.tight_layout()
# plt.savefig("模型对比科研图.png", bbox_inches='tight') # 取消注释可保存
plt.show()