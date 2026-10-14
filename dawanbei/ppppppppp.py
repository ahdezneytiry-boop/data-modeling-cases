import numpy as np
from sklearn.linear_model import LinearRegression


# 这里假设data是你的数据，格式为[[2020, value2020], [2021, value2021], [2022, value2022], [2023, value2023]]
data = np.array([[2020,6.67], [2021,7.46], [2022,6.68], [2023, 14]])

# 提取年份和经济数据
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

# 创建线性回归模型对象
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 创建要预测的年份数据（2024 - 2029）
future_years = np.array([[2024], [2025], [2026], [2027], [2028], [2029]]).reshape(-1, 1)

# 进行预测
predictions = model.predict(future_years)

print("预测结果：")
for year, prediction in zip(future_years, predictions):
    print(f"{year[0]}年的预测值: {prediction}")
