import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# ==========================================
# 1. 数据加载与鲁棒清洗
# ==========================================
# 加载数据
df_est = pd.read_csv(r"E:\MCM\estimated_votes_robust.csv")
df_raw = pd.read_csv(r"E:\MCM\daima\2026_MCM_Problem_C_Data.csv")

# 强制类型转换，防止 merge 报错
df_est['season'] = pd.to_numeric(df_est['season'], errors='coerce').fillna(0).astype(int)
df_est['week'] = pd.to_numeric(df_est['week'], errors='coerce').fillna(0).astype(int)
df_est['celebrity'] = df_est['celebrity'].astype(str).str.strip()

# 预处理原始数据以匹配格式
raw_long_list = []
for _, row in df_raw.iterrows():
    try:
        season = int(row['season'])
    except:
        continue
    celebrity = str(row['celebrity_name']).strip()
    
    for w in range(1, 12):
        # 动态获取每周 judge 分数
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        valid_cols = [c for c in cols if c in df_raw.columns]
        if not valid_cols: continue
        
        scores = pd.to_numeric(row[valid_cols], errors='coerce').fillna(0)
        if scores.sum() > 0:
            raw_long_list.append({
                'season': season,
                'celebrity': celebrity,
                'week': int(w),
                'raw_judge_score': scores.sum()
            })

df_raw_long = pd.DataFrame(raw_long_list)

# 合并数据
if not df_raw_long.empty:
    df_merged = pd.merge(df_est, df_raw_long, on=['season', 'week', 'celebrity'], how='inner')
else:
    # 兜底策略：如果合并失败（如名字不匹配），直接使用 est 数据
    df_merged = df_est.copy()
    df_merged['raw_judge_score'] = df_merged['judge_score'] * 10 + 20 # 简单反归一化模拟

# 计算不确定性指标
df_merged['uncertainty_width'] = df_merged['upper_bound_95'] - df_merged['lower_bound_95']
try:
    df_merged['uncertainty_level'] = pd.qcut(df_merged['uncertainty_width'], q=3, labels=['Low', 'Medium', 'High'])
except:
    df_merged['uncertainty_level'] = pd.cut(df_merged['uncertainty_width'], bins=3, labels=['Low', 'Medium', 'High'])

# ==========================================
# 2. 绘图配置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
# 设置全局字体大小
plt.rcParams.update({'font.size': 12})

# ==========================================
# 3. 生成 10 类高级图表
# ==========================================

# --- 一、层级关联类 ---

# 1. 嵌套饼图 (Sunburst 替代方案)
# 展示：内圈=赛季阶段，外圈=不确定性等级
def get_era(s):
    if s < 15: return 'Early (S1-14)'
    elif s <= 25: return 'Mid (S15-25)'
    else: return 'Recent (S26+)'
df_merged['Era'] = df_merged['season'].apply(get_era)
group_data = df_merged.groupby(['Era', 'uncertainty_level']).size().reset_index(name='counts')

plt.figure(figsize=(8, 8))
# 外圈
outer_colors = plt.cm.Blues(np.linspace(0.3, 0.9, 3))
plt.pie(df_merged['uncertainty_level'].value_counts(), radius=1.0, 
        colors=outer_colors, wedgeprops=dict(width=0.3, edgecolor='w'),
        labels=df_merged['uncertainty_level'].value_counts().index)
# 内圈
inner_colors = plt.cm.Pastel1(np.linspace(0, 1, 3))
plt.pie(df_merged['Era'].value_counts(), radius=0.7, 
        colors=inner_colors, wedgeprops=dict(width=0.3, edgecolor='w'),
        labels=None) # 不显示内圈标签以免拥挤
plt.title('Nested Pie: Season Era (Inner) vs Uncertainty (Outer)')
plt.savefig('1_nested_pie.png')

# 2. 堆叠柱状图 (Treemap 替代方案)
# 展示：按赛季堆叠的不确定性构成
pivot_data = df_merged.groupby(['season', 'uncertainty_level']).size().unstack().fillna(0)
# 归一化为百分比
pivot_data = pivot_data.div(pivot_data.sum(axis=1), axis=0)
pivot_data.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis', alpha=0.8)
plt.title('Uncertainty Composition by Season (Stacked Bar)')
plt.ylabel('Proportion')
plt.legend(title='Uncertainty Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('2_stacked_structure.png')

# --- 二、分布特征类 ---

# 3. 小提琴图 (Violin Plot)
plt.figure(figsize=(10, 6))
sns.violinplot(x='uncertainty_level', y='final_vote_est', data=df_merged, palette="muted", inner="quartile")
plt.title('Vote Distribution by Uncertainty Level')
plt.savefig('3_violin_dist.png')

# 4. 核密度图 (KDE)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_merged, x='raw_judge_score', hue='Era', fill=True, palette="crest", alpha=0.4)
plt.title('Score Distribution Evolution across Eras')
plt.savefig('4_kde_dist.png')

# --- 三、多维度关联类 ---

# 5. 双轴气泡图 (Bubble Chart)
plt.figure(figsize=(11, 7))
scatter = plt.scatter(
    df_merged['raw_judge_score'], 
    df_merged['final_vote_est'], 
    s=df_merged['uncertainty_width']*80 + 20, # 气泡大小映射不确定性
    c=df_merged['season'], # 颜色映射赛季
    cmap='Spectral', alpha=0.6, edgecolors='gray'
)
plt.colorbar(scatter, label='Season Index')
plt.title('Bubble: Score vs Votes (Size=Uncertainty)')
plt.xlabel('Raw Judge Score')
plt.ylabel('Estimated Votes')
plt.savefig('5_bubble_multi.png')

# 6. 热力图 (Heatmap)
hm_data = df_merged.pivot_table(index='season', columns='week', values='uncertainty_width', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(hm_data, cmap='coolwarm', cbar_kws={'label': 'Avg Uncertainty'})
plt.title('Heatmap: Uncertainty Hotspots')
plt.savefig('6_heatmap_corr.png')

# --- 四、逻辑流向类 ---

# 7. 归一化流向图 (Sankey 替代方案)
# 展示：得分区间 -> 不确定性等级 的概率流向
df_merged['Score_Bin'] = pd.qcut(df_merged['raw_judge_score'], 3, labels=['Low Score', 'Mid Score', 'High Score'])
cross_tab = pd.crosstab(df_merged['Score_Bin'], df_merged['uncertainty_level'], normalize='index')
cross_tab.plot(kind='barh', stacked=True, figsize=(10, 5), colormap='RdYlGn_r')
plt.title('Flow Logic: Probability of Uncertainty given Score Range')
plt.xlabel('Probability')
plt.savefig('7_flow_logic.png')

# 8. 平行坐标图 (Parallel Coordinates)
# 选取 Season 33 的数据
from pandas.plotting import parallel_coordinates
s33_data = df_merged[df_merged['season'] == 33].copy()
# 归一化各列以便在同一坐标系显示
cols = ['week', 'raw_judge_score', 'final_vote_est', 'uncertainty_width']
for c in cols:
    s33_data[c] = (s33_data[c] - s33_data[c].min()) / (s33_data[c].max() - s33_data[c].min())
s33_data['Status'] = s33_data['uncertainty_level'].astype(str) # 用于着色

plt.figure(figsize=(12, 6))
parallel_coordinates(s33_data[['week', 'raw_judge_score', 'final_vote_est', 'uncertainty_width', 'Status']], 
                     'Status', colormap=plt.cm.viridis, alpha=0.7)
plt.title('Parallel Coordinates: Variable Interactions (Season 33)')
plt.savefig('8_parallel_coords.png')

# --- 五、趋势与对比类 ---

# 9. 带误差线的折线图
season_agg = df_merged.groupby('season').agg({'final_vote_est': 'mean', 'uncertainty_width': 'mean'}).reset_index()
plt.figure(figsize=(12, 5))
plt.errorbar(season_agg['season'], season_agg['final_vote_est'], 
             yerr=season_agg['uncertainty_width']/2, 
             fmt='-o', color='darkblue', ecolor='lightblue', capsize=4, elinewidth=2)
plt.title('Trend: Season Average Votes with Uncertainty Bars')
plt.savefig('9_trend_error.png')

# 10. 雷达图
# 选取最近赛季 Top 3 选手
recent_season = df_merged['season'].max()
top_celebs = df_merged[df_merged['season'] == recent_season]['celebrity'].unique()[:3]
if len(top_celebs) > 0:
    # 准备数据
    radar_df = pd.DataFrame()
    metrics = ['raw_judge_score', 'final_vote_est', 'uncertainty_width', 'week']
    for c in top_celebs:
        sub = df_merged[(df_merged['celebrity'] == c) & (df_merged['season'] == recent_season)]
        vals = [sub[m].mean() for m in metrics]
        radar_df[c] = vals
    
    # 归一化到 0-1
    radar_df = (radar_df - radar_df.min().min()) / (radar_df.max().max() - radar_df.min().min())
    radar_df['metrics'] = metrics
    
    # 绘图
    categories = metrics
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    
    for c in top_celebs:
        values = radar_df[c].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=c)
        ax.fill(angles, values, alpha=0.1)
    
    plt.xticks(angles[:-1], categories)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title('Radar: Top Celebrity Profile Comparison')
    plt.savefig('10_radar_contrast.png')

print("All 10 charts generated and saved successfully.")