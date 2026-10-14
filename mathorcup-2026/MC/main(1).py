import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


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
    x = data[40:, 0]
    y = data[40:, 3:7]
    y_max = max(y.flatten())

    colors = ["#f6b014", "#7674b9", "#90abd4", "#b25f76"]
    # 绘制折线图
    for i in range(y.shape[1]):
        ax.step(x, y[:, i], where="mid",label=label[i], color=colors[i], linewidth=3)
    
    # ax.step(x, y, where="mid", label=label, linewidth=3)

    # 添加阴影区域
    ax.fill_betweenx([0, y_max * 1.1], 2016, 2020, color=colors[3], alpha=0.1)
    ax.plot([2016, 2016], [0, y_max * 1.1], color=colors[3], linestyle="--")
    ax.text(2016 - 0.2, y_max * 0.9, "2016年", ha="right", va="bottom", color=colors[3])
    ax.plot([2020, 2020], [0, y_max * 1.1], color=colors[3], linestyle="--")
    ax.text(2020 - 0.2, y_max * 0.9, "2020年", ha="right", va="bottom", color=colors[3])

    # 设置标题和轴标签
    ax.set_xlabel("年份", loc="right", labelpad=-12)  # X轴标签
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_ylim(0, y_max * 1.05)

    # 创建图例
    ax.legend(loc="center left", bbox_to_anchor=(0.1, 0.85), ncols=2)

    # 设置刻度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
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
