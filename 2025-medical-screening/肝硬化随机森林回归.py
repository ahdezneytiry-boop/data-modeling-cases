import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import joblib
import sklearn

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


file_path = "data/cirrhosis.csv"
data = pd.read_csv(file_path, encoding='utf-8')


print(f"数据形状: {data.shape}")
print(f"数据列名: {data.columns.tolist()}")
print(f"数据前5行:\n{data.head()}")


print("\n各列数据类型:")
print(data.dtypes)

print("\n分类变量唯一值:")
categorical_cols = ['Status', 'Ascites', 'Hepatomegaly', 'Spiders']
for col in categorical_cols:
    print(f"{col}: {data[col].unique()}")


from sklearn.preprocessing import LabelEncoder


label_encoders = {}


for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("\n预处理后数据前5行:")
print(data.head())


X = data[['N_Days', 'Status', 'Ascites', 'Hepatomegaly', 'Spiders', 
          'Bilirubin', 'Albumin', 'Copper', 'Platelets', 'Prothrombin', 'Alk_Phos']]
y = data['Stage']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  
    shuffle=False,  
    random_state=42  
)

print(f"scikit-learn版本: {sklearn.__version__}")


if sklearn.__version__ < '1.0':
    valid_criteria = ['mse', 'mae']
    criterion = 'mse'
    print(f"合法的criterion值: {valid_criteria}")
    print("使用criterion='mse'（旧版本兼容）")
else:
    valid_criteria = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
    criterion = 'squared_error'
    print(f"合法的criterion值: {valid_criteria}")
    print("使用criterion='squared_error'（新版本推荐）")

# 构建模型
rf = RandomForestRegressor(
    n_estimators=100,  
    criterion=criterion,  
    max_depth=10, 
    min_samples_split=2,  
    min_samples_leaf=1,  
    min_weight_fraction_leaf=0.0,  
    max_features=None,  
    max_leaf_nodes=50,  
    min_impurity_decrease=0.0, 
    bootstrap=True, 
    oob_score=False,  
    random_state=42
)


rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    '特征名称': X.columns,
    '特征重要性': [f'{round(imp*100, 1)}%' for imp in rf.feature_importances_]
})
print("\n===== 特征重要性 =====")
print(feature_importance.sort_values(by='特征重要性', ascending=False))


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
        'MAPE(%)': [round(mape, 3)],
        'R²': [round(r2, 3)]
    })

y_train_pred = rf.predict(X_train)
train_eval = evaluate(y_train, y_train_pred, '训练集')


y_test_pred = rf.predict(X_test)
test_eval = evaluate(y_test, y_test_pred, '测试集')

print("\n===== 模型评估结果 =====")
print(pd.concat([train_eval, test_eval], ignore_index=True))


test_results = pd.DataFrame({
    '预测测试集结果Y': y_test_pred,
    'Stage(实际值)': y_test.values
})
test_results = pd.concat([test_results, X_test.reset_index(drop=True)], axis=1)
print("\n===== 测试集预测结果（前15条） =====")
print(test_results.head(15))


joblib.dump(rf, 'random_forest_model.pkl')
print("\n模型已保存为：random_forest_model.pkl")


plt.figure(figsize=(10, 6))
sorted_idx = rf.feature_importances_.argsort()

cmap = LinearSegmentedColormap.from_list('blue_gradient', ['lightblue', 'darkblue'])
norm = plt.Normalize(0, len(sorted_idx))


for i, idx in enumerate(sorted_idx):
    color = cmap(norm(i))
    plt.barh(X.columns[idx], rf.feature_importances_[idx], color=color)

plt.xlabel('特征重要性')
plt.title('肝硬化阶段预测模型特征重要性')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight', facecolor='w')
plt.show()