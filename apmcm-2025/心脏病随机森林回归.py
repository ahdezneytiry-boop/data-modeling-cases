import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import joblib
import sklearn


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


file_path =r"E:\2025年第十五届APMCM亚太地区大学生数学建模竞赛（中文赛项）\B题 疾病的预测与大数据分析\第一问\预处理后结果\姐\heart.csv" # 文档明确数据源
data = pd.read_csv(file_path, encoding='utf-8')


categorical_cols = ['ChestPainType', 'Sex', 'ExerciseA0gi0a', 'ST_Slope']
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


X = data[['ChestPainType', 'Age', 'Sex', 'ExerciseA0gi0a', 'Oldpeak', 'ST_Slope', 'MaxHR']]
y = data['HeartDisease']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,
    shuffle=False,
    random_state=42
)


criterion = 'mse' if sklearn.__version__ < '1.0' else 'squared_error'
rf = RandomForestRegressor(
    n_estimators=100,
    criterion=criterion,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    max_leaf_nodes=50,
    bootstrap=True,
    oob_score=False,
    random_state=42
)


rf.fit(X_train, y_train)


y_train_pred = rf.predict(X_train)  
y_test_pred = rf.predict(X_test)    

feature_importance = pd.DataFrame({
    '特征名称': ['ChestPainType', 'Age', 'Sex', 'ExerciseA0gi0a', 'Oldpeak', 'ST_Slope', 'MaxHR'],
    '特征重要性': [f'{round(imp*100, 1)}%' for imp in rf.feature_importances_]
})
print("===== 特征重要性 =====")
print(feature_importance)


def evaluate(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    y_true_nonzero = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100
    r2 = r2_score(y_true, y_pred)
    return pd.DataFrame({
        '数据集': [dataset_name],
        'MSE': [round(mse, 3)],
        'RMSE': [round(rmse, 3)],
        'MAE': [round(mae, 3)],
        'MAPE': [round(mape, 3)],
        'R²': [round(r2, 3)]
    })

train_eval = evaluate(y_train, y_train_pred, '训练集') 
test_eval = evaluate(y_test, y_test_pred, '测试集')     
print("\n===== 模型评估结果 =====")
print(pd.concat([train_eval, test_eval], ignore_index=True))


test_results = pd.DataFrame({
    '预测测试集结果Y': y_test_pred,
    'HeartDisease': y_test.values
})
test_results = pd.concat([test_results, X_test.reset_index(drop=True)], axis=1)
print("\n===== 测试集预测结果（前15条） =====")
print(test_results.head(15))

# 11. 特征重要性图（保持原图形形式）
plt.figure(figsize=(10, 6))
sorted_idx = rf.feature_importances_.argsort()
cmap = LinearSegmentedColormap.from_list('blue_gradient', ['lightblue', 'darkblue'])
norm = plt.Normalize(0, len(sorted_idx))
for i, idx in enumerate(sorted_idx):
    color = cmap(norm(i))
    plt.barh(X.columns[idx], rf.feature_importances_[idx], color=color)
plt.xlabel('特征重要性')
plt.title('各特征（自变量）的重要性比例')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()