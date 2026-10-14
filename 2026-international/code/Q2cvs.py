import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ==========================================
# 1. 数据准备
# ==========================================
def load_and_prep_data():
    # 读取数据
    try:
        df_main = pd.read_csv("data/merged_data_for_q2.csv")
        df_meta = pd.read_csv("data/daima/2026_MCM_Problem_C_Data.csv")
    except:
        print("请确保当前目录下有 merged_data_for_q2.csv 和 2026_MCM_Problem_C_Data.csv")
        return None

    # 名称清洗与合并
    def clean(x): return str(x).lower().strip().replace(" ", "")
    df_main['key'] = df_main['clean_name'].apply(clean)
    df_meta['key'] = df_meta['celebrity_name'].apply(clean)
    
    # 合并：获取职业和淘汰信息
    df = df_main.merge(df_meta[['key', 'celebrity_industry', 'placement', 'results']], on='key', how='left')
    
    # 填充缺失值
    df['celebrity_industry'] = df['celebrity_industry'].fillna('Unknown')
    
    # 计算爆米花指数
    scaler = StandardScaler()
    df['norm_vote'] = scaler.fit_transform(df[['final_vote_est']])
    df['norm_score'] = scaler.fit_transform(df[['judge_score']])
    df['popcorn_index'] = df['norm_vote'] - df['norm_score']
    
    return df

# ==========================================
# 2. 绘制 3D 生存轨迹图
# ==========================================
def plot_3d_survival_trajectory(df):
    print("正在生成 3D 职业生存轨迹图...")
    
    # 筛选主要职业（样本量最多的前6个，避免图太乱）
    top_occs = df['celebrity_industry'].value_counts().head(6).index.tolist()
    df_sub = df[df['celebrity_industry'].isin(top_occs)]
    
    # 建立 Y 轴映射 (Occupation -> Int)
    occ_map = {name: i for i, name in enumerate(top_occs)}
    
    # 设置画布风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 配色方案
    colors = sns.color_palette("husl", len(top_occs))
    
    # 遍历每个选手画线
    for celeb_name, group in df_sub.groupby('clean_name'):
        group = group.sort_values('week_int')
        
        # 获取该选手的职业索引
        occ_name = group['celebrity_industry'].iloc[0]
        y_idx = occ_map[occ_name]
        
        # 坐标数据
        x = group['week_int'].values
        y = np.full_like(x, y_idx) # Y轴固定为职业ID
        z = group['popcorn_index'].values
        
        # 1. 画轨迹线
        ax.plot(x, y, z, color=colors[y_idx], alpha=0.4, linewidth=1.5)
        
        # 2. 画终点状态 (淘汰 vs 决赛)
        last_point = group.iloc[-1]
        
        # 判断是否被淘汰
        # 逻辑：如果 results 包含 'Eliminated' 或 placement > 3
        is_eliminated = False
        res_str = str(last_point['results']).lower()
        place_str = str(last_point['placement']).lower()
        
        if 'eliminated' in res_str or 'eliminated' in place_str:
            is_eliminated = True
        elif 'place' in place_str:
            # 提取名次数字
            try:
                rank = int(''.join(filter(str.isdigit, place_str.split()[0])))
                if rank > 3: is_eliminated = True
            except:
                pass
        
        # 绘制终点标记
        if is_eliminated:
            # 被淘汰：画红色 'x'
            ax.scatter(x[-1], y_idx, z[-1], marker='x', color='red', s=80, linewidth=2, zorder=10)
        else:
            # 存活/决赛：画绿色星星
            ax.scatter(x[-1], y_idx, z[-1], marker='*', color='gold', s=100, edgecolors='black', zorder=10)

    # 美化坐标轴
    ax.set_xlabel('Competition Week', fontsize=12, labelpad=10)
    ax.set_ylabel('Occupation', fontsize=12, labelpad=10)
    ax.set_zlabel('Popcorn Index (Vote - Score)', fontsize=12, labelpad=10)
    
    # 设置 Y 轴刻度为职业名称
    ax.set_yticks(range(len(top_occs)))
    ax.set_yticklabels(top_occs, rotation=-15, va='center', ha='left', fontsize=10)
    
    # 添加图注
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', label='Eliminated ', markerfacecolor='red', markersize=10, markeredgewidth=2, markeredgecolor='red'),
        Line2D([0], [0], marker='*', color='w', label='Finalist ', markerfacecolor='gold', markersize=15, markeredgecolor='black'),
        Line2D([0], [0], color='grey', lw=2, label='Survival Path ')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    ax.set_title('3D Survival Trajectories of Contestants by Occupation', fontsize=16)
    ax.view_init(elev=30, azim=-60) # 调整视角以便观察
    
    plt.tight_layout()
    plt.savefig('publication_plots/3D_Survival_Trajectory.png', dpi=300)
    print("绘图完成！已保存为 'publication_plots/3D_Survival_Trajectory.png'")

# 运行
if __name__ == "__main__":
    df = load_and_prep_data()
    if df is not None:
        plot_3d_survival_trajectory(df)