import numpy as np
import matplotlib.pyplot as plt


class GreyPrediction:
    def __init__(self, data):
        self.data = np.array(data)

    def predict(self, n):
        # 1. 累加生成
        accum_data = self.accumulate(self.data)

        # 2. 计算系数
        n_data = len(self.data)
        B = np.array([[-0.5 * (accum_data[i] + accum_data[i + 1]), 1] for i in range(n_data - 1)])
        Y = self.data[1:].reshape(-1, 1)
        a, b = np.linalg.inv(B.T @ B) @ B.T @ Y

        # 3. 预测
        forecast = []
        x0_prev = self.data[0]
        for i in range(n_data, n_data + n):
            x1_prev = (x0_prev - b / a) * np.exp(-a * i) + b / a
            forecast.append(x1_prev)
            x0_prev = x1_prev

        return forecast

    def accumulate(self, data):
        accum = [data[0]]
        for i in range(1, len(data)):
            accum.append(accum[i - 1] + data[i])
        return np.array(accum)


# 示例数据（可替换为实际的经济发展因素数据）
gdp_data = [1000, 1200, 1400, 1600, 1800]
import_export_data = [800, 900, 1000, 1100, 1200]
fixed_asset_investment_data = [500, 600, 700, 800, 900]

gp_gdp = GreyPrediction(gdp_data)
gp_import_export = GreyPrediction(import_export_data)
gp_fixed_asset = GreyPrediction(fixed_asset_investment_data)

forecast_gdp = gp_gdp.predict(5)
forecast_import_export = gp_import_export.predict(5)
forecast_fixed_asset = gp_fixed_asset.predict(5)

# 绘制预测结果
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(range(1, len(gdp_data) + 1), gdp_data, label='Original GDP')
plt.plot(range(len(gdp_data), len(gdp_data) + len(forecast_gdp)), forecast_gdp, label='Forecast GDP')
plt.xlabel('Time')
plt.ylabel('GDP')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(1, len(import_export_data) + 1), import_export_data, label='Original Import-Export')
plt.plot(range(len(import_export_data), len(import_export_data) + len(forecast_import_export)), forecast_import_export,
         label='Forecast Import-Export')
plt.xlabel('Time')
plt.ylabel('Import-Export')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(range(1, len(fixed_asset_investment_data) + 1), fixed_asset_investment_data,
         label='Original Fixed Asset Investment')
plt.plot(range(len(fixed_asset_investment_data), len(fixed_asset_investment_data) + len(forecast_fixed_asset)),
         forecast_fixed_asset, label='Forecast Fixed Asset Investment')
plt.xlabel('Time')
plt.ylabel('Fixed Asset Investment')
plt.legend()

plt.tight_layout()
plt.show()
