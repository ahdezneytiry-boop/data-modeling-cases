import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# ====================== 基础配置 ======================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# ====================== 数据加载与预处理 ======================
def load_dancing_data():
    df = pd.read_csv("data/processed_dancing_data_final.csv", encoding="utf-8-sig")
    # 计算平均得分（排除0分）
    score_cols = [col for col in df.columns if "week" in col and "score" in col]
    df["avg_score"] = df[score_cols].replace(0, np.nan).mean(axis=1).fillna(0)
    # 简化比赛结果
    df["result_simple"] = df["results"].apply(lambda x: 
        "1st Place" if "1st" in x else
        "2nd Place" if "2nd" in x else
        "3rd Place" if "3rd" in x else
        "Withdrew" if "Withdrew" in x else "Eliminated"
    )
    # 过滤有效数据
    df = df[df["celebrity_industry"].notna() & (df["avg_score"] > 0)]
    return df

df = load_dancing_data()

# ====================== 1. 六边形热力图（年龄 vs 平均得分） ======================
def plot_hexbin_heatmap(figsize=(10,6)):
    x = df["celebrity_age_during_season"].values
    y = df["avg_score"].values
    z = df["avg_score"].values

    colors = ["#2c3e50", "#3498db", "#ecf0f1", "#f39c12", "#e74c3c"]
    cmap = LinearSegmentedColormap.from_list("dance_cmap", colors, N=256)

    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(
        x, y, C=z, gridsize=30, cmap=cmap,
        reduce_C_function=np.mean, edgecolors="white", linewidths=0.8
    )
    cb = fig.colorbar(hb, ax=ax, orientation="vertical", pad=0.01, fraction=0.05)
    cb.set_label("平均得分", fontsize=12)
    
    ax.set_title("舞蹈比赛选手年龄与平均得分分布", fontsize=14, pad=20, fontweight="bold")
    ax.set_xlabel("选手年龄", fontsize=12)
    ax.set_ylabel("平均得分", fontsize=12)
    ax.set_xlim(15, 75)
    ax.set_ylim(0, 10)
    plt.tight_layout()
    plt.savefig("hexbin_heatmap.png", bbox_inches="tight", facecolor="white")
    plt.show()

# ====================== 2. 小提琴+箱线图（不同行业得分分布） ======================
def plot_violin_boxplot(figsize=(12,7)):
    # 筛选top5行业
    top_industries = df["celebrity_industry"].value_counts().head(5).index.tolist()
    industry_data = [df[df["celebrity_industry"] == ind]["avg_score"].values 
                     for ind in top_industries]
    
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    fig, ax = plt.subplots(figsize=figsize)
    
    # 小提琴图
    parts = ax.violinplot(industry_data, positions=range(len(top_industries)), 
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("white")
        pc.set_alpha(0.3)
    
    # 箱线图
    bp = ax.boxplot(industry_data, positions=range(len(top_industries)), widths=0.2,
                    patch_artist=True, showfliers=True, flierprops=dict(marker="o", markersize=4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(2)
    
    ax.set_title("不同行业选手平均得分分布", fontsize=14, pad=20, fontweight="bold")
    ax.set_xlabel("行业", fontsize=12)
    ax.set_ylabel("平均得分", fontsize=12)
    ax.set_xticks(range(len(top_industries)))
    ax.set_xticklabels(top_industries, rotation=15, ha="right")
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("violin_boxplot.png", bbox_inches="tight", facecolor="white")
    plt.show()

# ====================== 3. 3D曲面图（周数 vs 评委打分） ======================
def plot_3d_surface(figsize=(12,8)):
    # 提取前10周平均打分
    week_nums = list(range(1, 11))
    judge1_scores = [df[f"week{i}_judge1_score"].replace(0, np.nan).mean() for i in week_nums]
    judge2_scores = [df[f"week{i}_judge2_score"].replace(0, np.nan).mean() for i in week_nums]
    
    # 构建网格
    X, Y = np.meshgrid(week_nums, np.linspace(4, 10, 50))
    Z = np.zeros_like(X)
    for i in range(len(week_nums)):
        Z[:,i] = judge2_scores[i] + 0.1*(Y[:,i] - judge1_scores[i])
    
    # 绘图
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    colorscale = LinearSegmentedColormap.from_list("scale", ["#1a237e", "#7986cb", "#e8eaf6"], N=256)
    surf = ax.plot_surface(X, Y, Z, cmap=colorscale, edgecolor="white", linewidth=0.5, alpha=0.8)
    
    # 底部投影
    ax.contour(X, Y, Z, zdir="z", offset=Z.min()-0.5, cmap=colorscale, alpha=0.5)
    
    ax.set_title("各周评委1与评委2打分关系3D曲面", fontsize=14, pad=30, fontweight="bold")
    ax.set_xlabel("比赛周数", fontsize=12, labelpad=10)
    ax.set_ylabel("评委1平均得分", fontsize=12, labelpad=10)
    ax.set_zlabel("评委2平均得分", fontsize=12, labelpad=10)
    ax.set_xlim(1, 10)
    ax.set_ylim(4, 10)
    ax.set_zlim(Z.min()-0.5, Z.max()+0.5)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()
    plt.savefig("3d_surface.png", bbox_inches="tight", facecolor="white")
    plt.show()

# ====================== 4. 3D立方体塔图（国家+行业得分） ======================
def plot_3d_cube_towers(figsize=(12,8)):
    # 数据聚合
    top_countries = df["celebrity_homecountry_region"].value_counts().head(3).index.tolist()
    top_industries = df["celebrity_industry"].value_counts().head(3).index.tolist()
    agg_data = df[df["celebrity_homecountry_region"].isin(top_countries) & 
                  df["celebrity_industry"].isin(top_industries)].groupby(
        ["celebrity_homecountry_region", "celebrity_industry"])["avg_score"].mean().reset_index()
    
    # 坐标映射
    x_pos = {c: i for i, c in enumerate(top_countries)}
    y_pos = {i: j for j, i in enumerate(top_industries)}
    xx = [x_pos[c] for c in agg_data["celebrity_homecountry_region"]]
    yy = [y_pos[i] for i in agg_data["celebrity_industry"]]
    zz = agg_data["avg_score"].values
    max_z = zz.max()
    
    # 绘制立方体函数
    def draw_cube(ax, x, y, z, size=0.2, color="#3498db"):
        vertices = np.array([
            [x-size, y-size, 0], [x+size, y-size, 0],
            [x+size, y+size, 0], [x-size, y+size, 0],
            [x-size, y-size, z], [x+size, y-size, z],
            [x+size, y+size, z], [x-size, y+size, z]
        ])
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        for edge in edges:
            ax.plot3D([vertices[edge[0],0], vertices[edge[1],0]],
                      [vertices[edge[0],1], vertices[edge[1],1]],
                      [vertices[edge[0],2], vertices[edge[1],2]],
                      color="white", linewidth=2)
        faces = [[0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]
        for face in faces:
            ax.plot3D([vertices[f,0] for f in face+[face[0]]],
                      [vertices[f,1] for f in face+[face[0]]],
                      [vertices[f,2] for f in face+[face[0]]],
                      color=color, alpha=0.6)
    
    # 绘图
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    # 颜色映射
    cmap = plt.cm.RdYlBu_r
    for x, y, z in zip(xx, yy, zz):
        color = cmap(z/max_z)
        draw_cube(ax, x, y, z, size=0.2, color=color)
    
    ax.set_title("不同国家/地区+行业选手平均得分3D塔图", fontsize=14, pad=30, fontweight="bold")
    ax.set_xlabel("国家/地区", fontsize=12, labelpad=10)
    ax.set_ylabel("行业", fontsize=12, labelpad=10)
    ax.set_zlabel("平均得分", fontsize=12, labelpad=10)
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(top_countries, rotation=15)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(top_industries)
    ax.set_zlim(0, max_z+1)
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_z))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1, label="平均得分")
    plt.tight_layout()
    plt.savefig("3d_cube_towers.png", bbox_inches="tight", facecolor="white")
    plt.show()

# ====================== 5. 嵌套饼图（替代旭日图：结果→行业分布） ======================
def plot_nested_pie(figsize=(10,10)):
    # 构建层级数据
    top_results = ["1st Place", "2nd Place", "3rd Place", "Eliminated"]
    top_industries = df["celebrity_industry"].value_counts().head(3).index.tolist()
    
    # 外层：比赛结果分布
    outer_data = [len(df[df["result_simple"] == res]) for res in top_results]
    outer_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    
    # 内层：冠军行业分布
    inner_data = [len(df[(df["result_simple"] == "1st Place") & 
                        (df["celebrity_industry"] == ind)]) for ind in top_industries]
    inner_colors = ["#9b59b6", "#1abc9c", "#f1c40f"]
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    # 外层饼图
    wedges, texts, autotexts = ax.pie(outer_data, labels=top_results, colors=outer_colors,
                                     radius=1, wedgeprops=dict(width=0.3, edgecolor="white"),
                                     autopct="%1.1f%%", pctdistance=0.85)
    # 内层饼图
    wedges2, texts2, autotexts2 = ax.pie(inner_data, labels=top_industries, colors=inner_colors,
                                        radius=0.7, wedgeprops=dict(width=0.3, edgecolor="white"),
                                        autopct="%1.1f%%", pctdistance=0.65)
    
    ax.set_title("比赛结果分布（外层）与冠军行业分布（内层）", fontsize=14, pad=20, fontweight="bold")
    # 添加图例
    outer_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(outer_colors, top_results)]
    inner_patches = [mpatches.Patch(color=color, label=f"冠军-{label}") for color, label in zip(inner_colors, top_industries)]
    ax.legend(handles=outer_patches+inner_patches, loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.savefig("nested_pie.png", bbox_inches="tight", facecolor="white")
    plt.show()

# ====================== 主函数（调用入口） ======================
if __name__ == "__main__":
    plot_hexbin_heatmap()          # 1. 年龄-得分六边形热力图
    plot_violin_boxplot()         # 2. 行业得分小提琴+箱线图
    plot_3d_surface()            # 3. 评委打分3D曲面图
    plot_3d_cube_towers()        # 4. 国家-行业得分3D塔图
    plot_nested_pie()             # 5. 结果-冠军行业嵌套饼图（替代旭日图）