import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 样式与参数
DATA_FILE = 'data.csv'          # 数据文件
FIGSIZE = (8, 6)                # 图尺寸
DPI = 140                       # 分辨率
COLOR_LINE = '#c0627a'          # 主线颜色
COLOR_BAND = '#c0627a'          # 区间基色
LINEWIDTH = 2.2                 # 主线宽度
EDGE_COLOR = '#d599a8'          # 区间边界线颜色
EDGE_LINEWIDTH = 1.1            # 区间边界线宽
# 四层区间的透明度（外→内）
ALPHAS = [0.10, 0.18, 0.26, 0.34]
USE_GRID = True

# 坐标轴范围
XLIM = (0, 10)
YLIM = (0, 200)

# 样式文件
try:
    plt.style.use("line.mplstyle")
except Exception:
    pass
# 字体设置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
df = pd.read_csv(DATA_FILE)

x = df['x'].values
med = df['median'].values
q05 = df['q05'].values
q10 = df['q10'].values
q20 = df['q20'].values
q30 = df['q30'].values
q70 = df['q70'].values
q80 = df['q80'].values
q90 = df['q90'].values
q95 = df['q95'].values
mask_fcst = df['is_forecast'].astype(bool).values

# 作图
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

if mask_fcst.any():
    x_f = x[mask_fcst]

    # 层定义：[(下界数组, 上界数组, 图例名)]
    layers = [
        (q05[mask_fcst], q95[mask_fcst], '区间1'),
        (q10[mask_fcst], q90[mask_fcst], '区间2'),
        (q20[mask_fcst], q80[mask_fcst], '区间3'),
        (q30[mask_fcst], q70[mask_fcst], '区间4'),
    ]

    # 填充
    for i, (low, up, name) in enumerate(layers):
        ax.fill_between(x_f, low, up, color=COLOR_BAND,
                        alpha=ALPHAS[i], linewidth=0, label=name, zorder=2)

    # 边界线
    for low, up, _ in layers:
        ax.plot(x_f, low, color=EDGE_COLOR, lw=EDGE_LINEWIDTH, zorder=3)
        ax.plot(x_f, up,  color=EDGE_COLOR, lw=EDGE_LINEWIDTH, zorder=3)

# 中位数主线
ax.plot(x, med, color=COLOR_LINE, lw=LINEWIDTH, label='数据1', zorder=4)

# 坐标与网格
if XLIM is not None:
    ax.set_xlim(*XLIM)
if YLIM is not None:
    ax.set_ylim(*YLIM)
if USE_GRID:
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.25)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(frameon=False, loc='upper left')

plt.tight_layout()
plt.show()