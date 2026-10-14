# ============================================
# 🎯 问题二：无信息泄露版本（真实预测模型）
# ============================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

# ============================================
# 路径
# ============================================

data_path = "data/附件1：样例数据(1).xlsx"
output_dir = r"C:\Users\ahdezneytiry-boop1\Desktop\MC\无泄露模型结果"
os.makedirs(output_dir, exist_ok=True)

# ============================================
# Step 1：读取数据
# ============================================

data = pd.read_excel(data_path)
data.columns = data.columns.str.strip()

data.rename(columns={
    "高血脂症二分类标签": "label",
    "痰湿质": "tan",
    "活动量表总分（ADL总分+IADL总分）": "activity",
}, inplace=True)

# ============================================
# Step 2：删除信息泄露变量（关键🔥）
# ============================================

leak_cols = [
    "TG（甘油三酯）",
    "LDL-C（低密度脂蛋白）",
    "HDL-C（高密度脂蛋白）",
    "TC（总胆固醇）",
    "血脂异常分型标签（确诊病例）",
    "样本ID"
]

data = data.drop(columns=[c for c in leak_cols if c in data.columns])

# ============================================
# Step 3：选择预测变量（人工筛选）
# ============================================

feature_cols = [
    "tan", "气虚质", "血瘀质",
    "activity", "ADL总分", "IADL总分",
    "年龄组", "性别", "吸烟史", "饮酒史",
    "血尿酸", "BMI", "空腹血糖"
]

feature_cols = [f for f in feature_cols if f in data.columns]

X = data[feature_cols]
y = data["label"]

print("最终预测特征：")
print(feature_cols)

# ============================================
# Step 4：缺失值处理
# ============================================

imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ============================================
# Step 5：划分训练 / 测试集
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# Step 6：模型对比（训练集）
# ============================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LR": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000))
    ]),
    "RF": RandomForestClassifier(n_estimators=200, random_state=42),
    "LGBM": lgb.LGBMClassifier(random_state=42)
}

results = []

for name, model in models.items():
    cv_res = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "recall"],
        n_jobs=1
    )

    results.append({
        "模型": name,
        "AUC均值": np.mean(cv_res["test_roc_auc"]),
        "F1均值": np.mean(cv_res["test_f1"]),
        "Recall均值": np.mean(cv_res["test_recall"])
    })

df_results = pd.DataFrame(results)
print("\n模型对比：")
print(df_results)

df_results.to_excel(os.path.join(output_dir, "模型对比_无泄露.xlsx"), index=False)

# ============================================
# Step 7：最终模型评估（测试集）
# ============================================

best_name = df_results.sort_values(by="AUC均值", ascending=False).iloc[0]["模型"]
best_model = models[best_name]

best_model.fit(X_train, y_train)

y_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n测试集结果：")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred_prob)
print("测试集AUC:", auc)

# ============================================
# Step 8：输出数据
# ============================================

report = X_test.copy()
report["真实标签"] = y_test.values
report["预测概率"] = y_pred_prob

report.to_excel(os.path.join(output_dir, "测试集预测结果_无泄露.xlsx"), index=False)

print("\n无信息泄露模型运行完成")
# ============================================
# 🎯 Step 9：风险分层（核心）
# ============================================

# 用测试集概率做分层
prob = y_pred_prob

# 统计阈值（分位数法，更稳定）
T1 = np.percentile(prob, 30)
T2 = np.percentile(prob, 70)

print("\n风险分层阈值：")
print("T1（低→中）=", T1)
print("T2（中→高）=", T2)

def risk_level(p):
    if p < T1:
        return "低风险"
    elif p < T2:
        return "中风险"
    else:
        return "高风险"

risk_labels = [risk_level(p) for p in prob]

# ============================================
# Step 10：医学修正（关键🔥）
# ============================================

result = X_test.copy()
result["预测概率"] = prob
result["基础风险"] = risk_labels

def medical_adjust(row):
    risk = row["基础风险"]

    # 痰湿强影响
    if "tan" in row and row["tan"] >= 60:
        if risk == "低风险":
            risk = "中风险"
        elif risk == "中风险":
            risk = "高风险"

    # 极端风险
    if "tan" in row and "activity" in row:
        if row["tan"] >= 80 and row["activity"] < 40:
            risk = "高风险"

    return risk

result["最终风险"] = result.apply(medical_adjust, axis=1)

print("\n风险分布：")
print(result["最终风险"].value_counts())

# ============================================
# Step 11：输出完整数据（你画图用🔥）
# ============================================

# 1️⃣ 每个样本详细结果
result.to_excel(os.path.join(output_dir, "完整风险分层数据.xlsx"), index=False)

# 2️⃣ 分层统计
result["最终风险"].value_counts().to_excel(
    os.path.join(output_dir, "风险人数统计.xlsx")
)

# 3️⃣ 各变量均值对比（画图神器🔥）
group_stats = result.groupby("最终风险").mean(numeric_only=True)
group_stats.to_excel(os.path.join(output_dir, "分层均值对比.xlsx"))

# 4️⃣ 各变量分布（画箱线图用）
result.describe().to_excel(os.path.join(output_dir, "变量分布.xlsx"))

# ============================================
# Step 12：特征组合挖掘（核心🔥）
# ============================================

from mlxtend.frequent_patterns import fpgrowth, association_rules

# 只分析高风险人群
high_risk = result[result["最终风险"] == "高风险"]

# 离散化（简单分箱）
disc = pd.DataFrame()

for col in ["tan", "activity", "BMI", "血尿酸"]:
    if col in high_risk.columns:
        disc[col] = pd.qcut(high_risk[col], 3, labels=["低", "中", "高"], duplicates="drop")

# 转one-hot
disc = pd.get_dummies(disc)

# 挖规则
freq = fpgrowth(disc, min_support=0.2, use_colnames=True)

rules = association_rules(freq, metric="confidence", min_threshold=0.7)

rules = rules.sort_values(by="lift", ascending=False)

rules.to_excel(os.path.join(output_dir, "高风险特征组合.xlsx"))

print("\n特征组合挖掘完成")

# ============================================
# Step 13：痰湿体质重点分析（题目核心🔥）
# ============================================

tan_analysis = result.groupby("最终风险")["tan"].mean()
tan_analysis.to_excel(os.path.join(output_dir, "痰湿分层分析.xlsx"))

print("\n所有分析完成")