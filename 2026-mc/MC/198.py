import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from functools import lru_cache

# =============================================================================
# 1. 环境配置与全局参数
# =============================================================================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径设置（请根据实际情况修改）
DATA_PATH = "data/附件1：样例数据(1).xlsx"
OUTPUT_DIR = r"C:\Users\ahdezneytiry-boop1\Desktop\MC\问题3输出结果_教科书版"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 干预成本参数
TCM_COST = {1: 30, 2: 80, 3: 130}  # 中医调理月成本
EX_COST = {1: 3, 2: 5, 3: 8}      # 运动单次成本
EX_TIME = {1: 10, 2: 20, 3: 30}   # 运动时长
MONTHS = 6
BUDGET_LIMIT = 2000               # 总预算约束

# =============================================================================
# 2. 数据预处理与样本筛选
# =============================================================================
print("Step 1: 正在进行数据清洗与特征工程...")
df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

# 统一关键列名
rename_dict = {
    "高血脂症二分类标签": "label",
    "痰湿质": "tan_score",
    "活动量表总分（ADL总分+IADL总分）": "activity",
}
df.rename(columns=rename_dict, inplace=True)

# 缺失值智能插补 (KNN方法)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

# 筛选：痰湿质(体质标签=5) 且 确诊高血脂(label=1)
tan_patients = df[(df["体质标签"] == 5) & (df["label"] == 1)].copy()
print(f"筛选完成，共计 {len(tan_patients)} 名符合条件的干预对象。")

# =============================================================================
# 3. 状态转移校准 (拟合回归模型)
# =============================================================================
print("Step 2: 正在拟合状态转移基准模型...")
fit_feats = ["activity", "年龄组", "BMI", "血尿酸", "空腹血糖", "性别"]
X_fit = tan_patients[fit_feats]
y_fit = tan_patients["tan_score"]

reg = LinearRegression().fit(X_fit, y_fit)
coef_df = pd.DataFrame({"变量": ["截距"] + fit_feats, "系数": [reg.intercept_] + list(reg.coef_)})
coef_df.to_excel(os.path.join(OUTPUT_DIR, "状态转移拟合参数.xlsx"), index=False)

# =============================================================================
# 4. 高级干预模型函数 (学术级逻辑)
# =============================================================================
def get_intervention_effect(score, activity, age_group, level, intensity, freq, month):
    """
    计算干预后的积分下降和依从性
    引入边际递减效应：积分越低，下降越难
    """
    # 依从性衰减模型
    compliance = 1.0 - 0.07*(intensity-1) - 0.02*max(freq-5, 0) - 0.03*max(age_group-2, 0)
    if activity < 40: compliance -= 0.1
    compliance = max(0.6, min(1.0, compliance))
    
    # 效果系数 (引入非线性缩放)
    # 基础下降率
    base_rate = 0.03*(intensity-1) + 0.01*max(freq-5, 0) + {1:0.006, 2:0.015, 3:0.025}[level]
    if activity >= 60: base_rate += 0.005
    
    # 边际效应：随着积分降低，干预效率下降
    marginal_factor = 1.0 / (1.0 + np.exp(-(score - 35) / 8))
    
    drop = score * base_rate * compliance * marginal_factor
    return max(0.2, drop), compliance

# =============================================================================
# 5. 动态规划求解引擎 (高性能优化版)
# =============================================================================
@lru_cache(None)
def optimize_path(month, s_curr, a_curr, age_group):
    """
    使用递归+备忘录实现动态规划
    目标：在预算内最小化 6 个月后的痰湿积分
    """
    if month == MONTHS:
        return s_curr, []

    best_score = float('inf')
    best_plan = []
    
    # 决策剪枝：基于积分选择推荐调理等级
    level = 1 if s_curr <= 58 else (2 if s_curr <= 61 else 3)
    
    # 决策剪枝：基于能力选择运动强度
    intensities = [1]
    if age_group <= 2 and a_curr >= 60: intensities = [1, 2, 3]
    elif age_group <= 4 and a_curr >= 40: intensities = [1, 2]

    for intensity in intensities:
        for freq in [3, 5, 7, 9]: # 代表性采样加速计算
            m_cost = TCM_COST[level] + 4 * freq * EX_COST[intensity]
            drop, comp = get_intervention_effect(s_curr, a_curr, age_group, level, intensity, freq, month)
            
            s_next = max(0, int(round(s_curr - drop)))
            a_next = min(100, int(round(a_curr + (1.0 if freq >= 5 else 0))))
            
            final_s, tail_path = optimize_path(month + 1, s_next, a_next, age_group)
            
            # 加入预算软约束：如果超过每月平均预算(333元)，增加微小惩罚
            total_val = final_s + (0.001 * m_cost if m_cost > 333 else 0)
            
            if total_val < best_score:
                best_score = total_val
                best_plan = [{
                    "月份": month + 1, "月初积分": s_curr, "调理等级": level,
                    "活动强度": intensity, "每周频次": freq, "月成本": m_cost,
                    "预期降分": round(drop, 2), "依从性": round(comp, 2), "月末积分": s_next
                }] + tail_path
                
    return best_score, best_plan

# =============================================================================
# 6. 批量生成与结果分析
# =============================================================================
print("Step 3: 正在为所有患者生成最优干预方案...")
all_summaries = []
target_ids = [1, 2, 3]
id_plans = {}

for _, row in tan_patients.iterrows():
    optimize_path.cache_clear() # 每个患者独立计算
    pid = int(row["样本ID"])
    
    _, plan = optimize_path(0, int(round(row["tan_score"])), int(round(row["activity"])), int(row["年龄组"]))
    
    if plan:
        p_df = pd.DataFrame(plan)
        p_df["样本ID"] = pid
        p_df["累计成本"] = p_df["月成本"].cumsum()
        
        # 统计摘要
        all_summaries.append({
            "样本ID": pid,
            "初始积分": row["tan_score"],
            "最终积分": p_df.iloc[-1]["月末积分"],
            "总成本": p_df["月成本"].sum(),
            "平均依从性": p_df["依从性"].mean(),
            "主流方案": f"等级{int(p_df['调理等级'].mode()[0])}+强度{int(p_df['活动强度'].mode()[0])}"
        })
        
        if pid in target_ids:
            id_plans[pid] = p_df

# 保存汇总报表
summary_final = pd.DataFrame(all_summaries)
summary_final.to_excel(os.path.join(OUTPUT_DIR, "全体患者干预优化结果摘要.xlsx"), index=False)

# =============================================================================
# 7. 个体可视化导出 (ID 1/2/3)
# =============================================================================
print("Step 4: 正在导出重点样本分析图表...")
for pid, plan_df in id_plans.items():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 积分曲线
    color1 = '#e16259'
    ax1.set_xlabel('月份 (Months)')
    ax1.set_ylabel('痰湿积分 (Tan-Score)', color=color1, fontweight='bold')
    lns1 = ax1.plot(plan_df["月份"], plan_df["月初积分"], 'o-', color=color1, label='积分变化', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 成本曲线
    ax2 = ax1.twinx()
    color2 = '#084594'
    ax2.set_ylabel('累计成本 (Cumulative Cost)', color=color2, fontweight='bold')
    lns2 = ax2.fill_between(plan_df["月份"], plan_df["累计成本"], alpha=0.2, color=color2, label='累计支出')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f"样本 ID {pid} 六个月优化干预路径分析", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    plan_df.to_excel(os.path.join(OUTPUT_DIR, f"ID_{pid}_详细干预方案.xlsx"), index=False)
    plt.savefig(os.path.join(OUTPUT_DIR, f"ID_{pid}_趋势分析图.png"), dpi=300)
    plt.close()

print(f"全部任务完成！请在目录 '{OUTPUT_DIR}' 中查看结果。")