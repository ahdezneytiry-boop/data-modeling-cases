import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline


def load_style():
    try:
        plt.style.use("line.mplstyle")
    except:
        pass
    # 设置中文字体，以正常显示中文
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False


def create_chart(ax):
    # 读取数据
    df = pd.read_csv("data.csv")
    data = df.to_numpy()
    label = df.columns[3:7]
    x = data[:, 0]
    y = data[:, 3:7]
    y_max = max(y.flatten())

    colors = ["#f6b014", "#7674b9", "#90abd4", "#b25f76"]
    # 绘制平滑折线图
    x_smooth = np.linspace(x.min(), x.max(), 300)
    for i in range(y.shape[1]):
        spl = make_interp_spline(x, y[:, i], k=3)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, label=label[i], linewidth=3, color=colors[i])

    # 添加阴影区域
    ax.fill_betweenx([0, y_max * 1.1], 2000, 2014, color=colors[3], alpha=0.1)
    ax.plot([2000, 2000], [0, y_max * 1.1], color=colors[3], linestyle="--")
    ax.text(2000 - 0.5, y_max * 0.9, "2000年", ha="right", va="bottom", color=colors[3])
    ax.plot([2014, 2014], [0, y_max * 1.1], color=colors[3], linestyle="--")
    ax.text(2014 + 0.5, y_max * 0.9, "2014年", ha="left", va="bottom", color=colors[3])
    ax.text(2007, y_max * 0.9, "全球化时期", ha="center", va="bottom", color=colors[3])

    # 添加标注
    ax.scatter(
        x[33],
        y[33, 3],
        color=colors[3],
        marker="o",
        s=100,
        edgecolors="white",
        linewidths=2,
        zorder=5,
    )
    ax.text(
        x[33], y[33, 3] * 1.05, "历史高位", ha="center", va="bottom", color=colors[3]
    )

    # 设置轴标签
    ax.set_xlabel("年份", loc="right", labelpad=-12)  # X轴标签
    ax.set_xlim(min(x), max(x) + 2)
    ax.set_ylim(0, y_max * 1.1)

    # 创建图例
    ax.legend(loc="center left", bbox_to_anchor=(0.1, 0.75), ncols=1)

    # 设置刻度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    # 格式化刻度
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: "%1.0f亿" % (x * 1e-8))
    )

    # 设置网格线
    ax.grid(which="major", axis="y")
    # 设置axes边框
    ax.spines["top"].set_visible(False)  # 隐藏顶部边框
    ax.spines["right"].set_visible(False)  # 隐藏右侧边框
    ax.spines["left"].set_visible(False)  # 隐藏左侧边框

    plt.tight_layout()


if __name__ == "__main__":
    load_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    create_chart(ax)
    plt.show()
