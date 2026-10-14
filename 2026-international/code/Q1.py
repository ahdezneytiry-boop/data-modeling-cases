import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心配置与参数
# ==========================================
FILE_PATH = "data/processed_dancing_data_final.csv"  # 请确保文件路径正确
OUTPUT_FILE = r"estimated_votes_robust.csv"

# GBR 参数 (增加抗过拟合参数)
GBR_PARAMS = {
    'n_estimators': 200,          # 增加树的数量，靠早停控制
    'learning_rate': 0.05,
    'max_depth': 3,               # 限制深度，防止过拟合
    'subsample': 0.8,             # 每次只用80%数据建树 (Stochastic Gradient Boosting)
    'validation_fraction': 0.1,   # 预留验证集用于早停
    'n_iter_no_change': 10,       # 10次迭代无提升则停止
    'random_state': 42
}

# 敏感性测试的 C 值集合
C_VALUES_TO_TEST = [1.0, 10.0, 50.0, 100.0]

# ==========================================
# 2. 数据加载与预处理 (保持逻辑，增强稳健性)
# ==========================================
def load_and_process_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("[Error] 文件未找到，请检查路径。")
        return None

    # 解析 Result 得到存活周数
    def parse_results(res):
        if pd.isna(res): return 11
        res = str(res).lower()
        if "eliminated week" in res:
            try: return int(res.split("week")[-1].strip())
            except: return 1
        elif any(x in res for x in ["place", "winner", "runner-up"]): return 11
        elif "withdrew" in res: return 5 # 假设中途退赛平均在第5周
        return 11

    df['weeks_survived'] = df.apply(lambda x: parse_results(x['results']), axis=1)
    
    # 转为 Long Format (选手-周次 粒度)
    long_data = []
    for _, row in df.iterrows():
        cid = row['celebrity_name']
        survived = row['weeks_survived']
        season = row['season']
        
        for w in range(1, 12): 
            if w > survived: break
            
            # 获取当周评委分 (处理缺失值)
            cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols: continue
            
            scores = pd.to_numeric(row[valid_cols], errors='coerce').fillna(0)
            score_sum = scores.sum()
            if score_sum == 0: continue # 跳过无分数的记录
            
            long_data.append({
                'season': season,
                'week': w,
                'celebrity': cid,
                'industry': str(row['celebrity_industry']),
                'age': row['celebrity_age_during_season'],
                'judge_score': score_sum,
                'is_eliminated': (w == survived) and (w < 11) # 标记是否为淘汰周
            })
            
    return pd.DataFrame(long_data)

def feature_engineering(df_long):
    # One-Hot 编码职业
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ind_enc = enc.fit_transform(df_long[['industry']])
    ind_cols = [f"ind_{i}" for i in range(ind_enc.shape[1])]
    
    ind_df = pd.DataFrame(ind_enc, columns=ind_cols, index=df_long.index)
    
    # 归一化数值特征
    scaler = StandardScaler()
    num_cols = ['age', 'judge_score', 'week']
    df_long[num_cols] = scaler.fit_transform(df_long[num_cols])
    
    X = pd.concat([df_long[num_cols], ind_df], axis=1)
    return X, df_long

# ==========================================
# 3. 软约束优化核心 (Soft Constraint)
# ==========================================
def optimize_votes_with_constraints(initial_votes, df_season, C):
    """
    针对单赛季数据，进行软约束优化。
    目标：最小化 (修正值 - 初始值)^2 + C * 违反排名的惩罚
    """
    weeks = df_season['week'].unique()
    final_votes = initial_votes.copy()
    total_violation = 0
    
    # 按周次独立优化 (因为每周投票是独立的)
    for w in weeks:
        week_mask = df_season['week'] == w
        if not week_mask.any(): continue
        
        indices = np.where(week_mask)[0]
        # 获取本周数据
        v0 = initial_votes[indices]
        j_scores = df_season.loc[week_mask, 'judge_score'].values # 注意：这里用的是归一化后的特征值作为代替，实际应用原值更好，但为演示逻辑简化
        is_elim = df_season.loc[week_mask, 'is_eliminated'].values
        
        elim_idx = np.where(is_elim)[0]
        safe_idx = np.where(~is_elim)[0]
        
        if len(elim_idx) == 0 or len(safe_idx) == 0:
            continue
            
        # 定义局部目标函数
        def objective(v_new):
            # 1. 保持原预测的趋势 (L2 Loss)
            distortion = np.sum((v_new - v0)**2)
            
            # 2. 排名约束: Eliminated 总分 < Safe 总分
            # Total = Vote + Judge
            # 我们希望: (V_safe + J_safe) - (V_elim + J_elim) > margin
            # 违规: (V_elim + J_elim) - (V_safe + J_safe) > -margin
            # Slack = max(0, diff + margin)
            
            # 向量化计算所有 pair 的违规
            v_safe = v_new[safe_idx]
            v_elim = v_new[elim_idx]
            # 这里简单起见，假设 Judge Score 对 Total 的贡献系数为 1 (实际可调整)
            # 注意：传入的 j_scores 是归一化的，这里直接用即可
            score_safe = v_safe + j_scores[safe_idx]
            score_elim = v_elim + j_scores[elim_idx]
            
            margin = 0.01
            # 广播计算差值矩阵: (N_elim, 1) - (1, N_safe)
            diff = score_elim[:, None] - score_safe[None, :] 
            slacks = np.maximum(0, diff + margin)
            penalty = np.sum(slacks)
            
            return distortion + C * penalty

        # 求解
        res = minimize(objective, v0, method='L-BFGS-B')
        final_votes[indices] = res.x
        total_violation += res.fun - np.sum((res.x - v0)**2) # 记录纯惩罚项大小
        
    return final_votes, total_violation

# ==========================================
# 4. 主流程：LOSO 交叉验证与不确定性分析
# ==========================================
def main_pipeline():
    print("[1] 加载数据...")
    df = load_and_process_data(FILE_PATH)
    if df is None: return
    
    X_all, df = feature_engineering(df)
    
    # 构造代理目标 (Proxy Target)
    # 假设：评委分高 + 存活久 = 粉丝票高 (添加少量噪声防止过拟合)
    # 归一化评委分已经有了，只需还原尺度以便计算
    proxy_target = (df['judge_score'] * 0.5) + (df['week'] * 0.5) + np.random.normal(0, 0.1, len(df))
    
    seasons = df['season'].unique()
    results_summary = []
    
    print(f"[2] 开始留一赛季交叉验证 (共 {len(seasons)} 个赛季)...")
    
    # 存储所有预测结果用于最终合并
    full_predictions = pd.DataFrame()
    
    for test_season in seasons:
        # --- 划分训练/测试集 ---
        train_mask = df['season'] != test_season
        test_mask = df['season'] == test_season
        
        X_train, y_train = X_all[train_mask], proxy_target[train_mask]
        X_test = X_all[test_mask]
        
        if len(X_test) == 0: continue
        
        # --- 训练 GBR (均值模型) ---
        # 使用 Huber Loss 增强对异常值的鲁棒性
        gbr = GradientBoostingRegressor(loss='huber', **GBR_PARAMS)
        gbr.fit(X_train, y_train)
        pred_initial = gbr.predict(X_test)
        
        # --- 训练 Quantile GBR (不确定性区间) ---
        # 下界 (5%)
        gbr_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05, **GBR_PARAMS)
        gbr_lower.fit(X_train, y_train)
        pred_lower = gbr_lower.predict(X_test)
        
        # 上界 (95%)
        gbr_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95, **GBR_PARAMS)
        gbr_upper.fit(X_train, y_train)
        pred_upper = gbr_upper.predict(X_test)
        
        # --- 敏感性测试与软约束优化 ---
        # 我们在这个 Test Season 上寻找最佳的修正值
        # 这里演示取 C=50 (根据之前经验)，实际可循环 C_VALUES_TO_TEST 找最优
        
        df_test_subset = df[test_mask].copy()
        pred_optimized, penalty_score = optimize_votes_with_constraints(pred_initial, df_test_subset, C=50.0)
        
        # --- 鲁棒性检查 (Robustness Check) ---
        # 向测试集评委评分添加噪声，看排名是否剧烈波动
        X_test_noisy = X_test.copy()
        X_test_noisy['judge_score'] += np.random.normal(0, 0.5, len(X_test)) # 添加噪声
        pred_noisy = gbr.predict(X_test_noisy)
        robustness_corr = np.corrcoef(pred_initial, pred_noisy)[0,1]
        
        # 记录结果
        df_test_subset['initial_vote_pred'] = pred_initial
        df_test_subset['final_vote_est'] = pred_optimized
        df_test_subset['lower_bound_95'] = pred_lower
        df_test_subset['upper_bound_95'] = pred_upper
        df_test_subset['uncertainty_width'] = pred_upper - pred_lower
        
        full_predictions = pd.concat([full_predictions, df_test_subset])
        
        print(f"Season {test_season}: Robustness={robustness_corr:.3f}, Penalty={penalty_score:.2f}")

    # ==========================================
    # 5. 结果分析与输出
    # ==========================================
    print("\n[3] 最终分析报告")
    print("-" * 50)
    
    # 计算整体一致性 (有多少淘汰结果被正确复现)
    # 简单逻辑：在优化后的票数下，淘汰者的总分是否真的低于当周最低的幸存者
    # (这在 optimize 中已经作为目标，这里是验证)
    total_weeks = 0
    correct_weeks = 0
    
    for (s, w), group in full_predictions.groupby(['season', 'week']):
        elim = group[group['is_eliminated']]
        safe = group[~group['is_eliminated']]
        if len(elim) > 0 and len(safe) > 0:
            total_weeks += 1
            # 检查：淘汰者最高分 < 幸存者最低分 (严格一致性)
            # 注意：这里简化比较，实际题目是 Rank Sum，这里用分数近似
            if (elim['final_vote_est'] + elim['judge_score']).max() < (safe['final_vote_est'] + safe['judge_score']).min():
                correct_weeks += 1
    
    consistency_rate = correct_weeks / total_weeks if total_weeks > 0 else 0
    print(f"历史淘汰结果复现率 (Consistency Rate): {consistency_rate*100:.2f}%")
    print(f"平均预测不确定性区间宽度 (Avg 95% CI Width): {full_predictions['uncertainty_width'].mean():.4f}")
    
    # 导出结果
    out_cols = ['season', 'week', 'celebrity', 'judge_score', 'initial_vote_pred', 'final_vote_est', 'lower_bound_95', 'upper_bound_95']
    full_predictions[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"[4] 结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main_pipeline()