import numpy as np
import pandas as pd


# 1. 对原始数据进行累加生成操作 (AGO)
def AGO(data):
    agodata = [data[0]]
    for i in range(1, len(data)):
        agodata.append(agodata[i - 1] + data[i])
    return np.array(agodata)


# 2. 计算紧邻均值生成序列
def mean_gen(agodata):
    meandata = [agodata[0]]
    for i in range(1, len(agodata) - 1):
        meandata.append((agodata[i] + agodata[i + 1]) / 2)
    meandata.append(agodata[-1])
    return np.array(meandata)


# 3. 构建GM(1,1)模型的系数矩阵并求解参数
def GM11(data):
    agodata = AGO(data)
    meandata = mean_gen(agodata)
    B = np.vstack([-meandata[:-1], np.ones(len(meandata) - 1)]).T
    Y = data[1:].reshape((len(data) - 1, 1))
    # 使用最小二乘法求解参数
    u = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    a = u[0, 0]
    b = u[1, 0]

    # 预测函数
    def predict(k):
        return (data[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (k - 1))

    return predict


# 假设这里有一个包含多个影响因素数据的DataFrame（示例数据结构）
# 每列代表一个影响因素，每行代表一个时间点的数据
data_frame = pd.DataFrame({
    'factor1': [10, 12, 15, 18, 22],
    'factor2': [5, 6, 8, 10, 12],
    # 可以添加更多的影响因素列
})


# 对每个因素进行灰色预测分析
for column in data_frame.columns:
    factor_data = data_frame[column].values
    predict_func = GM11(factor_data)
    # 假设预测未来3个时间点
    for i in range(1, 4):
        prediction = predict_func(i)
        print(f'Factor: {column}, Prediction for time {i}: {prediction[0]}')
