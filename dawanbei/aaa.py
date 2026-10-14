# 灰色预测GM(1,1)模型
def GM11(x0):
    x1 = np.cumsum(x0)
    n = len(x1)
    z1 = np.zeros((n - 1,))
    for i in range(0, n - 1):
        z1[i] = -0.5 * (x1[i] + x1[i + 1])

    # 构建矩阵B和向量Y
    B = np.column_stack((z1, np.ones((n - 1, 1))))
    Y = x0[1:].reshape((n - 1, 1))
    # 计算参数向量[a,b]
    u = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    a = u[0][0]
    b = u[1][0]

    # 预测
    x_pred = np.zeros((n,))
    x_pred[0] = x0[0]
    for i in range(1, n):
        x_pred[i] = (x0[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (i))
    return x_pred


# 假设每个因素有10年的数据（这里只是示例数据，你需要根据实际情况替换）
population_total = np.array([100, 105, 110, 112, 115, 118, 120, 122, 125, 128])
age_structure = np.array([0.3, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4])
education_level = np.array([0.6, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7])
R_D_investment = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])
patent_application = np.array([500, 550, 600, 650, 700, 750, 800, 850, 900, 950])
high_tech_enterprise = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
industrial_structure = np.array([0.4, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5])
large_enterprise = np.array([500, 520, 540, 560, 580, 600, 620, 640, 660, 680])
strategic_industry_ratio = np.array([0.1, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])
transport_network_density = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
port_throughput = np.array([10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000])
information_level = np.array([0.7, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8])
import_export_total = np.array([5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500])
FDI = np.array([100, 120, 140, 160, 180, 200, 220, 240, 260, 280])
cross_border_flow = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])


# 对每个因素进行预测
predicted_population_total = GM11(population_total)
predicted_age_structure = GM11(age_structure)
predicted_education_level = GM11(education_level)
predicted_R_D_investment = GM11(R_D_investment)
predicted_patent_application = GM11(patent_application)
predicted_high_tech_enterprise = GM11(high_tech_enterprise)
predicted_industrial_structure = GM11(industrial_structure)
predicted_large_enterprise = GM11(large_enterprise)
predicted_strategic_industry_ratio = GM11(strategic_industry_ratio)
predicted_transport_network_density = GM11(transport_network_density)
predicted_port_throughput = GM11(port_throughput)
predicted_information_level = GM11(information_level)
predicted_import_export_total = GM11(import_export_total)
predicted_FDI = GM11(FDI)
predicted_cross_border_flow = GM11(cross_border_flow)


# 绘制预测结果（这里只是简单绘制每个因素的原始数据和预测数据对比，你可以根据需要进一步美化和分析）
plt.figure(figsize=(12, 8))

plt.subplot(4, 4, 1)
plt.plot(population_total, 'b-', label='Original')
plt.plot(predicted_population_total, 'r-', label='Predicted')
plt.title('Population Total')
plt.legend()

plt.subplot(4, 4, 2)
plt.plot(age_structure, 'b-', label='Original')
plt.plot(predicted_age_structure, 'r-', label='Predicted')
plt.title('Age Structure')
plt.legend()

plt.subplot(4, 4, 3)
plt.plot(education_level, 'b-', label='Original')
plt.plot(predicted_education_level, 'r-', label='Predicted')
plt.title('Education Level')
plt.legend()

plt.subplot(4, 4, 4)
plt.plot(R_D_investment, 'b-', label='Original')
plt.plot(predicted_R_D_investment, 'r-', label='Predicted')
plt.title('R&D Investment')
plt.legend()

plt.subplot(4, 4, 5)
plt.plot(patent_application, 'b-', label='Original')
plt.plot(predicted_patent_application, 'r-', label='Predicted')
plt.title('Patent Application')
plt.legend()

plt.subplot(4, 4, 6)
plt.plot(high_tech_enterprise, 'b-', label='Original')
plt.plot(predicted_high_tech_enterprise, 'r-', label='Predicted')
plt.title('High - Tech Enterprise')
plt.legend()

plt.subplot(4, 4, 7)
plt.plot(industrial_structure, 'b-', label='Original')
plt.plot(predicted_industrial_structure, 'r-', label='Predicted')
plt.title('Industrial Structure')
plt.legend()

plt.subplot(4, 4, 8)
plt.plot(large_enterprise, 'b-', label='Original')
plt.plot(predicted_large_enterprise, 'r-', label='Predicted')
plt.title('Large Enterprise')
plt.legend()

plt.subplot(4, 4, 9)
plt.plot(strategic_industry_ratio, 'b-', label='Original')
plt.plot(predicted_strategic_industry_ratio, 'r-', label='Predicted')
plt.title('Strategic Industry Ratio')
plt.legend()

plt.subplot(4, 4, 10)
plt.plot(transport_network_density, 'b-', label='Original')
plt.plot(predicted_transport_network_density, 'r-', label='Predicted')
plt.title('Transport Network Density')
plt.legend()

plt.subplot(4, 4, 11)
plt.plot(port_throughput, 'b-', label='Original')
plt.plot(predicted_port_throughput, 'r-', label='Predicted')
plt.title('Port Throughput')
plt.legend()

plt.subplot(4, 4, 12)
plt.plot(information_level, 'b-', label='Original')
plt.plot(predicted_information_level, 'r-', label='Predicted')
plt.title('Information Level')
plt.legend()

plt.subplot(4, 4, 13)
plt.plot(import_export_total, 'b-', label='Original')
plt.plot(predicted_import_export_total, 'r-', label='Predicted')
plt.title('Import - Export Total')
plt.legend()

plt.subplot(4, 4, 14)
plt.plot(FDI, 'b-', label='Original')
plt.plot(predicted_FDI, 'r-', label='Predicted')
plt.title('FDI')
plt.legend()

plt.subplot(4, 4, 15)
plt.plot(cross_border_flow, 'b-', label='Original')
plt.plot(predicted_cross_border_flow, 'r-', label='Predicted')
plt.title('Cross - Border Flow')
plt.legend()

plt.tight_layout()
plt.show()
