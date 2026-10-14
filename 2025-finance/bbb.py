import numpy as np
import matplotlib.pyplot as plt


# 灰色预测模型的GM(1,1)算法实现
def GM11(x0):
    x1 = np.cumsum(x0)
    n = len(x0)
    z1 = np.array([(x1[i] + x1[i - 1]) / 2 for i in range(1, n)])
    z1 = z1.reshape((n - 1, 1))
    B = np.hstack([-z1, np.ones((n - 1, 1))])
    Y = x0[1:].reshape((n - 1, 1))
    a, b = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y).flatten()

    def predict(k):
        return (x0[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (k - 1))

    return predict


# 假设这里是一些历史数据（示例数据，你需要替换为真实数据）
# 假设人口总量的历史数据
population_history = np.array([100, 105, 110])
# 年龄结构、受教育程度等其他因素数据结构类似

# 对人口总量进行预测
predict_population = GM11(population_history)
future_population = [predict_population(i) for i in range(len(population_history) + 1, len(population_history) + 6)]

# 对于其他影响因素也需要类似地进行数据准备和预测
#...

# 这里可以根据预测结果进行一些经济发展的综合评估（非常复杂，仅简单示意）
# 假设简单地认为人口总量和经济发展正相关
economic_prediction = [p * 1.2 for p in future_population]


# 绘制预测结果（仅人口总量预测结果示例）
plt.plot(list(range(len(population_history))), population_history, 'bo', label='History')
plt.plot(list(range(len(population_history), len(population_history) + len(future_population))), future_population,
         'r-', label='Future Prediction')
plt.title('Population Prediction')
plt.legend()
plt.show()
