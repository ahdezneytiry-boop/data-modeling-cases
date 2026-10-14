import numpy as np
import pandas as pd
# 灰色预测GM(1,1)模型
def GM11(x0):
    x1 = np.cumsum(x0)
    n = len(x1)
    z1 = np.array([(x1[i] + x1[i - 1]) / 2 for i in range(1, n)])
    B = np.vstack([-z1, np.ones(n - 1)]).T
    Y = x0[1:].reshape((n - 1, 1))
    # 计算参数
    u = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    a = u[0, 0]
    b = u[1, 0]
    # 预测公式
    x_pred = [(x0[0] - b / a) * np.exp(-a * i) + b / a for i in range(1, n + 1)]
    # 还原预测值
    x_pred = np.array(x_pred)
    x_pred[0] = x0[0]
    for i in range(1, n):
        x_pred[i] = x_pred[i] - x_pred[i - 1]
        return x_pred
    # 假设你的数据是一个DataFrame，每列是一个影响因素，行是时间序列数据
data = pd.DataFrame({
    'population': [100, 110, 120, 130, 140], # 人口总量示例数据
    'age_structure': [0.2, 0.22, 0.23, 0.25, 0.26], # 年龄结构示例数据
    'education_level': [0.6, 0.65, 0.7, 0.75, 0.8], # 受教育程度示例数据
    'R&D_investment': [10, 12, 15, 18, 20], # 研发投入示例数据
    'patent_application': [50, 60, 70, 80, 90], # 专利申请量示例数据
    'high_tech_company': [10, 15, 20, 25, 30], # 高新技术企业数量示例数据
    'industry_structure': [0.3, 0.32, 0.35, 0.38, 0.4], # 产业结构示例数据
    'scale_company': [500, 550, 600, 650, 700], # 规模以上企业数量示例数据
    'strategic_industry_ratio': [0.1, 0.12, 0.15, 0.18, 0.2], # 战略新兴产业占比示例数据
    'traffic_network_density': [0.5, 0.55, 0.6, 0.65, 0.7], # 交通网络密度示例数据
    'port_throughput': [1000, 1200, 1500, 1800, 2000], # 港口吞吐量示例数据
    'information_level': [0.4, 0.45, 0.5, 0.55, 0.6], # 信息化水平示例数据
    'import_export_total': [5000, 6000, 7000, 8000, 9000], # 进出口总额示例数据
    'FDI': [100, 120, 150, 180, 200], # 外商直接投资额示例数据
    'cross - border_flow': [50, 60, 70, 80, 90] # 跨境人员流动量示例数据
})
    # 对每个影响因素进行灰色预测
predicted_results = {}
for col in data.columns:
    x0 = np.array(data[col])
    prediction = GM11(x0)
    predicted_results[col] = prediction
    # 将预测结果转换为DataFrame
predicted_df = pd.DataFrame(predicted_results)
