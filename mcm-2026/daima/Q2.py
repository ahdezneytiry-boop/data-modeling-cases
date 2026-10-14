import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, spearmanr
import warnings

# ==========================================
# 0. 科研级绘图设置
# ==========================================
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300 

COLORS = {
    'Judge': "#6295b9",  # 蓝
    'Fan': "#965d5d",    # 红
    'Composite': "#488448" # 绿
}

# ==========================================
# 1. 数据加载与预处理
# ==========================================
def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
        
    # 填充缺失值
    df['raw_judge_score'] = df['raw_judge_score'].fillna(0)
    df['concrete_votes'] = df['concrete_votes'].fillna(0)
    
    # 统一名字格式，方便后续查找
    df['clean_name_lower'] = df['clean_name'].astype(str).str.lower().str.strip()
    
    return df

# ==========================================
# 2. 核心仿真引擎
# ==========================================
def simulate_week(season_df, week, method_type, alpha=0.5, save_mechanism=True, vote_noise=0.05):
    week_df = season_df[season_df['week_int'] == week].copy()
    if len(week_df) < 2: return None

    # 1. 引入噪声
    noise = np.random.uniform(1 - vote_noise, 1 + vote_noise, size=len(week_df))
    week_df['sim_votes'] = week_df['concrete_votes'] * noise
    
    j_scores = week_df['raw_judge_score'].values
    f_votes = week_df['sim_votes'].values
    
    # 2. 计算排名/得分
    if method_type == 'rank':
        # 排名法：分数越小越好 (1=第一名)
        # 用负号排序实现：高分->Rank 1
        r_judge = rankdata(-j_scores, method='min') 
        r_fan = rankdata(-f_votes, method='min')
        
        # 核心改动：排名法的加权逻辑
        # Alpha 是评委权重的占比
        total_score = alpha * r_judge + (1 - alpha) * r_fan
        
    elif method_type == 'pct':
        s_j = j_scores.sum() if j_scores.sum() > 0 else 1
        s_f = f_votes.sum() if f_votes.sum() > 0 else 1
        
        p_judge = (j_scores / s_j) * 100
        p_fan = (f_votes / s_f) * 100
        
        # 核心改动：百分比法的加权逻辑
        total_score = alpha * p_judge + (1 - alpha) * p_fan
        total_score = -total_score # 取反，保持"值越大越差"的排序逻辑
    
    week_df['final_score'] = total_score
    
    # 3. 排序
    week_df = week_df.sort_values(by=['final_score'], ascending=False)
    
    # 4. 评委救人 (Save)
    eliminated_idx = 0 
    if save_mechanism and len(week_df) >= 2:
        bottom_1 = week_df.iloc[0]
        bottom_2 = week_df.iloc[1]
        # 如果倒一比倒二强太多 (10% 阈值)，则救回
        if bottom_1['raw_judge_score'] > bottom_2['raw_judge_score'] * 1.10:
            eliminated_idx = 1
            
    eliminated_row = week_df.iloc[eliminated_idx]
    
    # 计算指标
    # 公平性：结果与评委分的相关性
    corr_j, _ = spearmanr(week_df['final_score'], -week_df['raw_judge_score'])
    # 满意度：结果与粉丝票的相关性
    corr_f, _ = spearmanr(week_df['final_score'], -week_df['sim_votes'])
    
    # 处理 NaN
    if np.isnan(corr_j): corr_j = 0
    if np.isnan(corr_f): corr_f = 0

    return {
        'season': week_df['season'].iloc[0],
        'week': week,
        'eliminated_name': eliminated_row['clean_name_lower'],
        'judge_correlation': corr_j,
        'fan_correlation': corr_f,
        'bobby_survived': 'bobby bones' not in eliminated_row['clean_name_lower']
    }

# ==========================================
# 3. 系统调参 (寻找 Golden Coefficient)
# ==========================================
def run_parameter_tuning(df, n_iterations=50): # 增加到 50 次
    print(f"开始系统调参 (蒙特卡洛迭代 {n_iterations} 次)...")
    
    # 扩大 Alpha 范围，更容易看到倒U型曲线
    alphas = np.linspace(0.1, 0.9, 9) 
    methods = ['rank', 'pct']
    
    tuning_log = []
    
    for m in methods:
        for a in alphas:
            fairness_scores = []  # Judge Correlation
            satisfaction_scores = [] # Fan Correlation
            
            for i in range(n_iterations):
                # 跑 S27 (典型赛季)
                s_data = df[df['season'] == 27]
                
                s_fairness = []
                s_satisfaction = []
                
                for w in sorted(s_data['week_int'].unique()):
                    res = simulate_week(s_data, w, m, alpha=a, save_mechanism=True)
                    if res:
                        s_fairness.append(res['judge_correlation'])
                        s_satisfaction.append(res['fan_correlation'])
                
                fairness_scores.append(np.mean(s_fairness))
                satisfaction_scores.append(np.mean(s_satisfaction))
            
            avg_fair = np.mean(fairness_scores)
            avg_sat = np.mean(satisfaction_scores)
            
            # 定义综合效用函数 (Composite Utility Function)
            # 我们给公平性更高权重 (0.7)，因为目标是修正"水货夺冠"
            composite_score = 0.7 * avg_fair + 0.3 * avg_sat
            
            tuning_log.append({
                'Method': 'Ranking' if m == 'rank' else 'Percentage',
                'Alpha': a,
                'Fairness (Judge)': avg_fair,
                'Satisfaction (Fan)': avg_sat,
                'Composite Score': composite_score
            })
            
    return pd.DataFrame(tuning_log)

# ==========================================
# 0. 优化的绘图设置 (Fix: Font Size & Layout)
# ==========================================
warnings.filterwarnings('ignore')
# 将 font_scale 从 1.2 降到 1.0，解决字体过大问题
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300 

COLORS = {
    'Judge': "#88c3ed",  # 蓝
    'Fan': "#b07272",    # 红
    'Composite': "#71a571" # 绿
}


def plot_tuning_results(tuning_df):
    """
    绘制双Y轴图：展示 Fairnes, Satisfaction 和 Composite Score
    优化：字体更小，布局更紧凑，防止标题遮挡
    """
    df_rank = tuning_df[tuning_df['Method'] == 'Ranking']
    
    # 使用 constrained_layout 自动防止遮挡
    fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # 绘制参考线 (虚线)
    l1, = ax1.plot(df_rank['Alpha'], df_rank['Fairness (Judge)'], color=COLORS['Judge'], 
             linestyle='--', alpha=0.6, label='Fairness (Judge Corr)')
    l2, = ax1.plot(df_rank['Alpha'], df_rank['Satisfaction (Fan)'], color=COLORS['Fan'], 
             linestyle='--', alpha=0.6, label='Satisfaction (Fan Corr)')
    
    ax1.set_xlabel('Judge Weight (Alpha)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Correlation Score', fontweight='bold', fontsize=11)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # 绘制主线 (实线)
    l3, = ax1.plot(df_rank['Alpha'], df_rank['Composite Score'], color=COLORS['Composite'], 
             linewidth=3, marker='o', markersize=8, label='Composite Utility Score')
    
    # 合并图例
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=9, frameon=True, framealpha=0.9)
    
    # 寻找并标注峰值
    best_idx = df_rank['Composite Score'].idxmax()
    best_row = df_rank.loc[best_idx]
    
    ax1.annotate(f"Optimal Alpha={best_row['Alpha']:.1f}\nScore={best_row['Composite Score']:.3f}", 
                 xy=(best_row['Alpha'], best_row['Composite Score']),
                 xytext=(best_row['Alpha'], best_row['Composite Score']+0.08),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', ha='center', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    # 标题调小一点，防止溢出
    plt.title('Parameter Tuning: Finding the "Golden Coefficient"', 
              fontsize=13, fontweight='bold', pad=15)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def plot_bobby_trajectory(df):
    """
    绘制 Bobby Bones 的排名轨迹
    优化：修复遮挡，调整 Gap 填充颜色，使其更美观
    """
    s27 = df[df['season'] == 27].copy()
    mask = s27['clean_name_lower'].str.contains('bobby', na=False)
    bobby_data = s27[mask]
    
    if bobby_data.empty:
        print("警告: 未在 S27 找到 Bobby Bones 的数据。")
        return

    weeks = sorted(bobby_data['week_int'].unique())
    judge_ranks = []
    fan_ranks = []
    
    for w in weeks:
        w_df = s27[s27['week_int'] == w]
        j_rank = rankdata(-w_df['raw_judge_score'], method='min')
        f_rank = rankdata(-w_df['concrete_votes'], method='min')
        
        w_df = w_df.reset_index(drop=True)
        b_indices = w_df.index[w_df['clean_name_lower'].str.contains('bobby', na=False)].tolist()
        
        if b_indices:
            idx = b_indices[0]
            judge_ranks.append(j_rank[idx])
            fan_ranks.append(f_rank[idx])
        else:
            judge_ranks.append(np.nan)
            fan_ranks.append(np.nan)
            
    # 使用 constrained_layout
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # 绘制轨迹
    plt.plot(weeks, judge_ranks, marker='o', markersize=8, color=COLORS['Judge'], 
             label='Judge Rank (Skill)', linewidth=2.5, linestyle='--')
    plt.plot(weeks, fan_ranks, marker='X', markersize=9, color=COLORS['Fan'], 
             label='Fan Vote Rank (Popularity)', linewidth=2.5)
    
    # 填充差异区域 (颜色更淡一点，防止遮挡网格)
    plt.fill_between(weeks, judge_ranks, fan_ranks, color='gray', alpha=0.15, hatch='///', label='Controversy Gap')
    
    # 反转Y轴
    plt.gca().invert_yaxis()
    plt.yticks(range(1, 14), fontsize=10)
    plt.xticks(weeks, fontsize=10)
    
    # 标题与标签
    plt.title('Performance Trajectory of "Controversial Star" (Bobby Bones, S27)', 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Competition Week', fontsize=11, fontweight='bold')
    plt.ylabel('Rank (1 = Top)', fontsize=11, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9, frameon=True)
    
    # 标注 Gap (字体调小，背景框优化)
    mid_week_idx = len(weeks) // 2
    mid_week = weeks[mid_week_idx]
    # 确保标注在Gap中间
    mid_val = (judge_ranks[mid_week_idx] + fan_ranks[mid_week_idx]) / 2
    
    plt.text(mid_week, mid_val, 'Huge Discrepancy\n(The Problem)', 
             ha='center', va='center', color='black', fontsize=9, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="black", alpha=0.95))
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    file_path = "merged_data_for_q2.csv"
    
    df = load_and_clean_data(file_path)
    
    if df is not None:
        # 1. 运行系统调参 (跑50次)
        tuning_results = run_parameter_tuning(df, n_iterations=50)
        print("\n=== 系统调参日志 (Top 5 Rows) ===")
        print(tuning_results.head())
        
        # 2. 绘制调参结果 (寻找峰值)
        plot_tuning_results(tuning_results)
        
        # 3. 绘制 Bobby Bones 轨迹 (修复版)
        plot_bobby_trajectory(df)