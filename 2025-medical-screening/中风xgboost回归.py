import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


data = pd.read_csv("data/stroke.csv", encoding='utf-8')  # 改为读取CSV文件
print("数据基本信息：")
print(f"数据形状: {data.shape}")
print(f"数据列名: {data.columns.tolist()}")
print("前5行数据:\n", data.head())

X = data[['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi']]
y = data['stroke']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    train_size=0.7, 
    shuffle=False,  
    random_state=42
)


model = XGBRegressor(
    objective='reg:squarederror',  # 回归任务
    booster='gbtree',  # 文档"基学习器：gbtree"
    n_estimators=100,  # 文档"基学习器数量：100"
    learning_rate=0.1,  # 文档"学习率：0.1"
    reg_alpha=0,  # 文档"L1正则项：0"
    reg_lambda=1,  # 文档"L2正则项：1"
    subsample=1,  # 文档"样本征采样率：1"
    colsample_bytree=1,  # 文档"树特征采样率：1"
    colsample_bylevel=1,  # 文档"节点特征采样率：1"
    min_child_weight=0,  # 文档"叶子节点中样本的最小权重：0"
    max_depth=10,  # 文档"树的最大深度：10"
    random_state=42
)

# 5. 训练模型
model.fit(X_train, y_train)
print(f"\n模型训练完成（训练用时参考：{0.55}s）")

# 6. 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 7. 模型评估
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
    'stroke': y_test.values
})
test_results = pd.concat([test_results, X_test.reset_index(drop=True)], axis=1)
print("\n===== 测试集预测结果（前15条） =====")
print(test_results.head(15))

feature_importance = pd.DataFrame({
    '特征名称': X.columns,
    '特征重要性': [f'{round(imp*100, 1)}%' for imp in model.feature_importances_]
})
print("\n===== 特征重要性 =====")
print(feature_importance)

plt.figure(figsize=(10, 6))
sorted_idx = model.feature_importances_.argsort()  # 按重要性排序


cmap = LinearSegmentedColormap.from_list('green_gradient', ['lightgreen', 'darkgreen'])
norm = plt.Normalize(0, len(sorted_idx))


for i, idx in enumerate(sorted_idx):
    color = cmap(norm(i))
    plt.barh(X.columns[idx], model.feature_importances_[idx], color=color)

plt.xlabel('特征重要性')
plt.title('各特征（自变量）的重要性比例')
plt.tight_layout()
plt.savefig('stroke_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='w')
plt.show()

# 11. 绘制测试数据预测图（真实值vs预测值）
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='真实值', marker='o', markersize=4, linestyle='-', linewidth=1)
plt.plot(y_test_pred[:100], label='预测值', marker='x', markersize=4, linestyle='--', linewidth=1)
plt.xlabel('样本索引')
plt.ylabel('stroke')
plt.title('测试数据预测图')
plt.legend()
plt.tight_layout()
plt.savefig('stroke_prediction_plot.png', dpi=300)
plt.show()