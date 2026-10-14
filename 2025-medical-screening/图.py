import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Arrow
import matplotlib.colors as mcolors

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12  # 整体文字放大

# 配色方案（更柔和的树状结构配色）
COLOR_ROOT = "#d1e7dd"       # 根节点（数据输入）
COLOR_PREP = "#cfe2ff"       # 预处理节点
COLOR_TREE = "#f8d7da"       # 树节点
COLOR_OUTPUT = "#f3d19e"     # 输出节点
COLOR_BORDER = "#333333"     # 边框
COLOR_ARROW = "#666666"      # 箭头
COLOR_TEXT = "#222222"       # 文字


def draw_tree_structure(model_name, dataset_name, is_boosting=False):
    """绘制树状结构模型图"""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    plt.title(f"{dataset_name} - {model_name}树状模型结构", 
              fontsize=18, pad=20, color=COLOR_TEXT, fontweight='bold')

    # 根节点：数据输入（顶层）
    root = FancyBboxPatch(
        (6, 8.5), 3, 1, boxstyle="round,pad=0.2", 
        fc=COLOR_ROOT, ec=COLOR_BORDER, lw=1.5
    )
    ax.add_patch(root)
    plt.text(7.5, 9, f"{dataset_name}", 
             ha='center', va='center', color=COLOR_TEXT, fontweight='bold')

    # 第二层：数据预处理
    prep = FancyBboxPatch(
        (6, 6.5), 3, 1, boxstyle="round,pad=0.2", 
        fc=COLOR_PREP, ec=COLOR_BORDER, lw=1.5
    )
    ax.add_patch(prep)
    plt.text(7.5, 7, "数据预处理", 
             ha='center', va='center', color=COLOR_TEXT)
    # 根节点到预处理的箭头
    ax.add_patch(Arrow(7.5, 8.5, 0, -1, width=0.15, color=COLOR_ARROW))

    # 第三层：模型核心（树结构）
    if not is_boosting:
        # 随机森林：多棵树并行（分支结构）
        tree1 = FancyBboxPatch(
            (2, 4.5), 2.5, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_TREE, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(tree1)
        plt.text(3.25, 5, "决策树1", ha='center', va='center', color=COLOR_TEXT)

        tree2 = FancyBboxPatch(
            (6, 4.5), 2.5, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_TREE, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(tree2)
        plt.text(7.25, 5, "决策树2", ha='center', va='center', color=COLOR_TEXT)

        tree3 = FancyBboxPatch(
            (10, 4.5), 2.5, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_TREE, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(tree3)
        plt.text(11.25, 5, "决策树3...", ha='center', va='center', color=COLOR_TEXT)

        # 预处理到各树的箭头（分支箭头）
        ax.add_patch(Arrow(7.5, 6.5, -3.5, -1, width=0.15, color=COLOR_ARROW))  # 左分支
        ax.add_patch(Arrow(7.5, 6.5, 0, -1, width=0.15, color=COLOR_ARROW))      # 中间分支
        ax.add_patch(Arrow(7.5, 6.5, 3.5, -1, width=0.15, color=COLOR_ARROW))   # 右分支

        # 第四层：集成节点
        ensemble = FancyBboxPatch(
            (6, 2.5), 3, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_TREE, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(ensemble)
        plt.text(7.5, 3, "结果集成", ha='center', va='center', color=COLOR_TEXT)

        # 各树到集成的箭头
        ax.add_patch(Arrow(3.25, 4.5, 3.25, -1, width=0.15, color=COLOR_ARROW))
        ax.add_patch(Arrow(7.25, 4.5, 0.25, -1, width=0.15, color=COLOR_ARROW))
        ax.add_patch(Arrow(11.25, 4.5, -3.75, -1, width=0.15, color=COLOR_ARROW))

        # 第五层：输出节点
        output = FancyBboxPatch(
            (6, 0.5), 3, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_OUTPUT, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(output)
        plt.text(7.5, 1, "预测结果", ha='center', va='center', color=COLOR_TEXT)
        # 集成到输出的箭头
        ax.add_patch(Arrow(7.5, 2.5, 0, -1, width=0.15, color=COLOR_ARROW))

    else:
        # XGBoost：迭代树链（链式结构）
        tree1 = FancyBboxPatch(
            (3, 4.5), 2.5, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_TREE, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(tree1)
        plt.text(4.25, 5, "初始树", ha='center', va='center', color=COLOR_TEXT)

        tree2 = FancyBboxPatch(
            (7, 4.5), 2.5, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_TREE, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(tree2)
        plt.text(8.25, 5, "修正树1", ha='center', va='center', color=COLOR_TEXT)

        tree3 = FancyBboxPatch(
            (11, 4.5), 2.5, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_TREE, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(tree3)
        plt.text(12.25, 5, "修正树2...", ha='center', va='center', color=COLOR_TEXT)

        # 预处理到第一棵树的箭头
        ax.add_patch(Arrow(7.5, 6.5, -3.25, -1, width=0.15, color=COLOR_ARROW))
        # 树之间的链式箭头
        ax.add_patch(Arrow(4.25, 4.5, 3, 0, width=0.15, color=COLOR_ARROW))
        ax.add_patch(Arrow(8.25, 4.5, 3, 0, width=0.15, color=COLOR_ARROW))

        # 最终输出节点
        output = FancyBboxPatch(
            (7, 2.5), 3, 1, boxstyle="round,pad=0.2", 
            fc=COLOR_OUTPUT, ec=COLOR_BORDER, lw=1.5
        )
        ax.add_patch(output)
        plt.text(8.5, 3, "预测结果", ha='center', va='center', color=COLOR_TEXT)
        # 最后一棵树到输出的箭头
        ax.add_patch(Arrow(12.25, 4.5, -3.75, -1, width=0.15, color=COLOR_ARROW))

    # 模型说明（右侧）
    notes = {
        "随机森林": "• 多棵决策树并行训练\n• 每棵树独立基于抽样数据\n• 结果通过投票/平均集成",
        "XGBoost回归": "• 串行迭代生成决策树\n• 每棵树修正前序误差\n• 带正则化控制复杂度"
    }
    plt.text(14, 7, f"{model_name}说明", 
             ha='center', va='center', color=COLOR_TEXT, fontweight='bold', fontsize=13)
    plt.text(14, 5, notes[model_name], 
             ha='center', va='top', color=COLOR_TEXT, fontsize=11, linespacing=1.6)

    plt.tight_layout()
    return fig


# 1. 心脏病数据集 - 随机森林（多树并行树状结构）
fig1 = draw_tree_structure(
    model_name="随机森林",
    dataset_name="心脏病数据集",
    is_boosting=False
)
plt.show()

# 2. 中风数据集 - XGBoost（迭代树链结构）
fig2 = draw_tree_structure(
    model_name="XGBoost回归",
    dataset_name="中风数据集",
    is_boosting=True
)
plt.show()

# 3. 肝硬化数据集 - 随机森林（多树并行树状结构）
fig3 = draw_tree_structure(
    model_name="随机森林",
    dataset_name="肝硬化数据集",
    is_boosting=False
)
plt.show()