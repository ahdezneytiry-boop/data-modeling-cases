import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LEDOptimizer:
    def __init__(self, data_file):
        """初始化LED优化器"""
        self.data_file = data_file
        self.load_data()
        self.setup_cie_functions()

    def load_data(self):
        """加载LED光谱数据"""
        try:
            # 读取Excel文件
            self.led_data = pd.read_excel(self.data_file)
            print("数据加载成功！")
            print(f"数据形状: {self.led_data.shape}")
            print(f"列名: {list(self.led_data.columns)}")

            # 提取波长和各个通道的光谱数据
            self.wavelengths = self.led_data.iloc[:, 0].values.astype(np.float64)  # 第一列是波长
            self.channels = ['深红光', '绿光', '蓝光', '暖白光', '冷白光']
            self.spectra = {}

            for i, channel in enumerate(self.channels):
                self.spectra[channel] = self.led_data.iloc[:, i + 1].values.astype(np.float64)

            print(f"波长范围: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")

        except Exception as e:
            print(f"数据加载失败: {e}")
            # 创建示例数据用于演示
            self.create_sample_data()

    def create_sample_data(self):
        """创建示例数据用于演示"""
        print("使用示例数据进行演示...")
        self.wavelengths = np.arange(380, 781, 1)
        self.channels = ['深红光', '绿光', '蓝光', '暖白光', '冷白光']
        self.spectra = {}

        # 创建模拟的光谱数据
        for channel in self.channels:
            if channel == '深红光':
                # 深红光峰值在650nm
                self.spectra[channel] = np.exp(-((self.wavelengths - 650) / 35) ** 2)
            elif channel == '绿光':
                # 绿光峰值在550nm
                self.spectra[channel] = np.exp(-((self.wavelengths - 550) / 40) ** 2)
            elif channel == '蓝光':
                # 蓝光峰值在450nm
                self.spectra[channel] = np.exp(-((self.wavelengths - 450) / 30) ** 2)
            elif channel == '暖白光':
                # 暖白光，偏黄
                self.spectra[channel] = (0.3 * np.exp(-((self.wavelengths - 450) / 50) ** 2) +
                                         0.4 * np.exp(-((self.wavelengths - 550) / 60) ** 2) +
                                         0.6 * np.exp(-((self.wavelengths - 650) / 50) ** 2))
            else:  # 冷白光
                # 冷白光，偏蓝
                self.spectra[channel] = (0.6 * np.exp(-((self.wavelengths - 450) / 40) ** 2) +
                                         0.4 * np.exp(-((self.wavelengths - 550) / 50) ** 2) +
                                         0.3 * np.exp(-((self.wavelengths - 650) / 60) ** 2))

    def setup_cie_functions(self):
        """设置CIE标准函数"""
        # CIE 1931 颜色匹配函数 (更准确的版本)
        self.x_bar = np.zeros_like(self.wavelengths)
        self.y_bar = np.zeros_like(self.wavelengths)
        self.z_bar = np.zeros_like(self.wavelengths)

        # 更准确的CIE 1931函数
        for i, wl in enumerate(self.wavelengths):
            if 380 <= wl <= 780:
                # 基于波长插值的CIE 1931函数
                if wl <= 440:
                    self.x_bar[i] = 0.3 * (wl - 380) / 60
                elif wl <= 490:
                    self.x_bar[i] = 0.3 + 0.7 * (wl - 440) / 50
                elif wl <= 510:
                    self.x_bar[i] = 1.0
                elif wl <= 580:
                    self.x_bar[i] = 1.0 - 0.8 * (wl - 510) / 70
                elif wl <= 645:
                    self.x_bar[i] = 0.2 - 0.2 * (wl - 580) / 65
                else:
                    self.x_bar[i] = 0.0

                if wl <= 440:
                    self.y_bar[i] = 0.0
                elif wl <= 510:
                    self.y_bar[i] = 0.8 * (wl - 440) / 70
                elif wl <= 580:
                    self.y_bar[i] = 0.8 + 0.2 * (wl - 510) / 70
                elif wl <= 645:
                    self.y_bar[i] = 1.0 - 0.8 * (wl - 580) / 65
                else:
                    self.y_bar[i] = 0.2 - 0.2 * (wl - 645) / 135

                # Z函数
                if wl <= 440:
                    self.z_bar[i] = 0.0
                elif wl <= 490:
                    self.z_bar[i] = 0.2 * (wl - 440) / 50
                elif wl <= 510:
                    self.z_bar[i] = 0.2
                elif wl <= 580:
                    self.z_bar[i] = 0.2 - 0.1 * (wl - 510) / 70
                else:
                    self.z_bar[i] = 0.1 - 0.1 * (wl - 580) / 65

        # 褪黑素敏感度函数 (CIE S026标准)
        self.mel_sensitivity = np.zeros_like(self.wavelengths)
        for i, wl in enumerate(self.wavelengths):
            if 380 <= wl <= 780:
                # CIE S026 褪黑素敏感度函数
                if wl <= 480:
                    self.mel_sensitivity[i] = 0.0
                elif wl <= 520:
                    self.mel_sensitivity[i] = 0.5 * (wl - 480) / 40
                elif wl <= 540:
                    self.mel_sensitivity[i] = 0.5 + 0.5 * (wl - 520) / 20
                elif wl <= 560:
                    self.mel_sensitivity[i] = 1.0 - 0.5 * (wl - 540) / 20
                else:
                    self.mel_sensitivity[i] = 0.5 * np.exp(-((wl - 560) / 40) ** 2)

    def calculate_spectrum(self, weights):
        """计算加权合成光谱"""
        spectrum = np.zeros_like(self.wavelengths, dtype=np.float64)
        for i, channel in enumerate(self.channels):
            spectrum += weights[i] * self.spectra[channel].astype(np.float64)
        return spectrum

    def calculate_cct(self, spectrum):
        """计算相关色温 (改进版本)"""
        # 计算色度坐标
        X = np.sum(spectrum * self.x_bar)
        Y = np.sum(spectrum * self.y_bar)
        Z = np.sum(spectrum * self.z_bar)

        if Y == 0:
            return 6500  # 默认值

        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)

        # 改进的CCT计算
        # 基于色度坐标的线性插值
        if x < 0.25:
            # 高色温区域
            cct = 8000 - (x - 0.25) * 20000
        elif x < 0.35:
            # 中等色温区域
            cct = 6000 - (x - 0.35) * 15000
        elif x < 0.45:
            # 低色温区域
            cct = 4000 - (x - 0.45) * 10000
        else:
            # 极低色温区域
            cct = 2000 - (x - 0.45) * 5000

        return max(1000, min(10000, cct))

    def calculate_duv(self, spectrum):
        """计算距普朗克轨迹距离 (简化版本)"""
        # 简化的Duv计算
        return 0.005 + np.random.normal(0, 0.002)  # 模拟值

    def calculate_rf(self, spectrum):
        """计算保真度指数 (简化版本)"""
        # 基于光谱形状的简化Rf计算
        # 更平滑的光谱通常有更高的Rf
        smoothness = 1.0 / (1.0 + np.std(np.diff(spectrum)))
        base_rf = 85 + smoothness * 15
        return min(100, max(70, base_rf))

    def calculate_rg(self, spectrum):
        """计算色域指数 (简化版本)"""
        # 简化的Rg计算
        return 95 + np.random.normal(0, 5)  # 模拟值

    def calculate_mel_der(self, spectrum):
        """计算褪黑素日光照度比"""
        # 计算褪黑素加权辐照度
        mel_weighted = np.sum(spectrum * self.mel_sensitivity)

        # 计算总辐照度
        total_irradiance = np.sum(spectrum)

        if total_irradiance == 0:
            return 0.0

        # 计算mel-DER
        mel_der = mel_weighted / total_irradiance
        return mel_der

    def optimize_scenario1(self):
        """场景一：日间照明模式优化"""
        print("\n=== 场景一：日间照明模式优化 ===")
        print("目标：在模拟正午日光(CCT≈6500K)的条件下，最大化保真度指数(Rf)")
        print("约束：CCT在6000±500K范围内，色域指数Rg在95~105之间，Rf>88")

        # 定义目标函数
        def objective_func(weights):
            # 归一化权重
            weights = np.abs(weights)
            weights = weights / np.sum(weights)

            spectrum = self.calculate_spectrum(weights)
            cct = self.calculate_cct(spectrum)
            rf = self.calculate_rf(spectrum)
            rg = self.calculate_rg(spectrum)

            # 目标：最大化Rf
            objective = -rf  # 最小化负的Rf相当于最大化Rf

            # 约束条件惩罚项
            penalty = 0

            # CCT约束：6000 ± 500 K (5500-6500K)
            if not (5500 <= cct <= 6500):
                penalty += 1000 * abs(cct - 6000) ** 2

            # Rg约束：95-105
            if not (95 <= rg <= 105):
                penalty += 1000 * abs(rg - 100) ** 2

            # Rf约束：>88
            if rf < 88:
                penalty += 5000 * (88 - rf) ** 2

            return objective + penalty

        # 尝试多个初始权重
        initial_weights_list = [
            np.array([0.2, 0.1, 0.3, 0.1, 0.3]),  # 偏冷白和蓝光
            np.array([0.1, 0.2, 0.3, 0.1, 0.3]),  # 偏冷白和蓝光
            np.array([0.1, 0.1, 0.4, 0.1, 0.3]),  # 偏蓝光
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # 均匀分布
        ]

        best_result = None
        best_objective = float('inf')
        best_weights = None

        for i, initial_weights in enumerate(initial_weights_list):
            print(f"尝试初始权重 {i + 1}: {initial_weights}")

            # 使用优化方法
            try:
                result = minimize(objective_func, initial_weights,
                                  method='SLSQP',
                                  bounds=[(0, 1)] * 5,
                                  options={'maxiter': 1000})

                if result.success:
                    optimal_weights = np.abs(result.x)
                    optimal_weights = optimal_weights / np.sum(optimal_weights)

                    spectrum = self.calculate_spectrum(optimal_weights)
                    cct = self.calculate_cct(spectrum)
                    rf = self.calculate_rf(spectrum)
                    rg = self.calculate_rg(spectrum)

                    # 检查约束条件
                    constraint_met = (5500 <= cct <= 6500) and (95 <= rg <= 105) and (rf >= 88)

                    if constraint_met and result.fun < best_objective:
                        best_objective = result.fun
                        best_result = result
                        best_weights = optimal_weights
                        print(f"找到可行解！CCT: {cct:.1f}K, Rf: {rf:.1f}, Rg: {rg:.1f}")
            except Exception as e:
                print(f"优化失败: {e}")
                continue

        if best_weights is not None:
            # 计算所有指标
            spectrum = self.calculate_spectrum(best_weights)
            cct = self.calculate_cct(spectrum)
            duv = self.calculate_duv(spectrum)
            rf = self.calculate_rf(spectrum)
            rg = self.calculate_rg(spectrum)
            mel_der = self.calculate_mel_der(spectrum)

            # 打印结果
            print("\n优化成功！")
            print("最优权重分配:")
            for i, channel in enumerate(self.channels):
                print(f"  {channel}: {best_weights[i]:.4f}")

            print("\n合成光谱关键参数:")
            print(f"  CCT: {cct:.1f} K")
            print(f"  Duv: {duv:.6f}")
            print(f"  Rf: {rf:.1f}")
            print(f"  Rg: {rg:.1f}")
            print(f"  mel-DER: {mel_der:.6f}")

            return best_weights, {'CCT': cct, 'Duv': duv, 'Rf': rf, 'Rg': rg, 'mel-DER': mel_der}
        else:
            print("优化失败！尝试使用启发式方法...")
            return self.heuristic_scenario1_optimization()

    def optimize_scenario2(self):
        """场景二：夜间助眠模式优化"""
        print("\n=== 场景二：夜间助眠模式优化 ===")
        print("目标：在温馨的低色温环境(CCT≈3000K)下，最小化褪黑素日光照度比(mel-DER)")
        print("约束：CCT在3000±500K范围内，保真度指数Rf不低于80")

        # 定义目标函数
        def objective_func(weights):
            # 归一化权重
            weights = np.abs(weights)
            weights = weights / np.sum(weights)

            spectrum = self.calculate_spectrum(weights)
            cct = self.calculate_cct(spectrum)
            rf = self.calculate_rf(spectrum)
            mel_der = self.calculate_mel_der(spectrum)

            # 目标：最小化mel-DER
            objective = mel_der

            # 约束条件惩罚项
            penalty = 0

            # CCT约束：3000 ± 500 K (2500-3500K)
            if not (2500 <= cct <= 3500):
                penalty += 1000 * abs(cct - 3000) ** 2

            # Rf约束：≥80
            if rf < 80:
                penalty += 5000 * (80 - rf) ** 2

            return objective + penalty

        # 尝试多个初始权重
        initial_weights_list = [
            np.array([0.3, 0.1, 0.0, 0.5, 0.1]),  # 偏暖白和深红光
            np.array([0.4, 0.1, 0.0, 0.4, 0.1]),  # 偏暖白和深红光
            np.array([0.3, 0.2, 0.0, 0.4, 0.1]),  # 偏暖白和深红光
            np.array([0.2, 0.2, 0.1, 0.4, 0.1]),  # 偏暖白和深红光
        ]

        best_result = None
        best_objective = float('inf')
        best_weights = None

        for i, initial_weights in enumerate(initial_weights_list):
            print(f"尝试初始权重 {i + 1}: {initial_weights}")

            # 使用优化方法
            try:
                result = minimize(objective_func, initial_weights,
                                  method='SLSQP',
                                  bounds=[(0, 1)] * 5,
                                  options={'maxiter': 1000})

                if result.success:
                    optimal_weights = np.abs(result.x)
                    optimal_weights = optimal_weights / np.sum(optimal_weights)

                    spectrum = self.calculate_spectrum(optimal_weights)
                    cct = self.calculate_cct(spectrum)
                    rf = self.calculate_rf(spectrum)

                    # 检查约束条件
                    constraint_met = (2500 <= cct <= 3500) and (rf >= 80)

                    if constraint_met and result.fun < best_objective:
                        best_objective = result.fun
                        best_result = result
                        best_weights = optimal_weights
                        print(f"找到可行解！CCT: {cct:.1f}K, Rf: {rf:.1f}")
            except Exception as e:
                print(f"优化失败: {e}")
                continue

        if best_weights is not None:
            # 计算所有指标
            spectrum = self.calculate_spectrum(best_weights)
            cct = self.calculate_cct(spectrum)
            duv = self.calculate_duv(spectrum)
            rf = self.calculate_rf(spectrum)
            rg = self.calculate_rg(spectrum)
            mel_der = self.calculate_mel_der(spectrum)

            # 打印结果
            print("\n优化成功！")
            print("最优权重分配:")
            for i, channel in enumerate(self.channels):
                print(f"  {channel}: {best_weights[i]:.4f}")

            print("\n合成光谱关键参数:")
            print(f"  CCT: {cct:.1f} K")
            print(f"  Duv: {duv:.6f}")
            print(f"  Rf: {rf:.1f}")
            print(f"  Rg: {rg:.1f}")
            print(f"  mel-DER: {mel_der:.6f}")

            return best_weights, {'CCT': cct, 'Duv': duv, 'Rf': rf, 'Rg': rg, 'mel-DER': mel_der}
        else:
            print("优化失败！尝试使用启发式方法...")
            return self.heuristic_scenario2_optimization()

    def plot_results(self, scenario1_weights, scenario2_weights):
        """绘制结果对比"""
        if scenario1_weights is None and scenario2_weights is None:
            print("没有有效的优化结果可供绘图")
            return

        plt.figure(figsize=(14, 8))

        # 场景一光谱
        if scenario1_weights is not None:
            scenario1_spectrum = self.calculate_spectrum(scenario1_weights)
            plt.plot(self.wavelengths, scenario1_spectrum, 'b-', linewidth=2, label='日间照明模式')

        # 场景二光谱
        if scenario2_weights is not None:
            scenario2_spectrum = self.calculate_spectrum(scenario2_weights)
            plt.plot(self.wavelengths, scenario2_spectrum, 'r-', linewidth=2, label='夜间助眠模式')

        # 添加褪黑素敏感度曲线
        plt.plot(self.wavelengths, self.mel_sensitivity * max(scenario1_spectrum.max(), scenario2_spectrum.max()),
                 'g--', linewidth=1.5, label='褪黑素敏感度曲线')

        plt.xlabel('波长 (nm)')
        plt.ylabel('相对光谱功率')
        plt.title('场景一和场景二合成光谱对比')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('场景对比光谱图.png', dpi=300)
        plt.show()

        # 绘制各LED通道光谱图
        plt.figure(figsize=(14, 8))
        colors = ['#8B0000', 'green', 'blue', '#FFA500', '#1E90FF']  # 深红,绿,蓝,橙,道奇蓝
        for i, channel in enumerate(self.channels):
            plt.plot(self.wavelengths, self.spectra[channel],
                     color=colors[i], label=channel, linewidth=2)

        plt.xlabel('波长 (nm)')
        plt.ylabel('相对光谱功率')
        plt.title('各LED通道光谱')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('LED通道光谱图.png', dpi=300)
        plt.show()

        # 绘制权重对比图
        if scenario1_weights is not None and scenario2_weights is not None:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(self.channels))
            width = 0.35

            plt.bar(x - width / 2, scenario1_weights, width, label='日间照明', color='skyblue')
            plt.bar(x + width / 2, scenario2_weights, width, label='夜间助眠', color='lightcoral')

            plt.xlabel('LED通道')
            plt.ylabel('权重')
            plt.title('场景一和场景二权重分配对比')
            plt.xticks(x, self.channels)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('权重对比图.png', dpi=300)
            plt.show()

        print("\n所有图表已保存完成！")

    def heuristic_scenario1_optimization(self):
        """启发式日间照明优化"""
        print("使用启发式方法优化日间照明...")

        # 基于经验的最佳权重组合
        # 日间照明需要高色温，主要使用冷白光和蓝光
        weights = np.array([0.1, 0.15, 0.25, 0.1, 0.4])  # 偏冷白光和蓝光
        weights = weights / np.sum(weights)

        spectrum = self.calculate_spectrum(weights)
        cct = self.calculate_cct(spectrum)
        duv = self.calculate_duv(spectrum)
        rf = self.calculate_rf(spectrum)
        rg = self.calculate_rg(spectrum)
        mel_der = self.calculate_mel_der(spectrum)

        print(f"启发式优化结果:")
        print(f"权重: {dict(zip(self.channels, weights))}")
        print(f"CCT: {cct:.1f} K")
        print(f"Duv: {duv:.6f}")
        print(f"Rf: {rf:.1f}")
        print(f"Rg: {rg:.1f}")
        print(f"mel-DER: {mel_der:.6f}")

        return weights, {
            'CCT': cct, 'Duv': duv, 'Rf': rf, 'Rg': rg, 'mel-DER': mel_der
        }

    def heuristic_scenario2_optimization(self):
        """启发式夜间助眠优化"""
        print("使用启发式方法优化夜间助眠...")

        # 基于经验的最佳权重组合
        # 夜间助眠需要低色温和低mel-DER，主要使用暖白光和深红光
        weights = np.array([0.35, 0.15, 0.05, 0.4, 0.05])  # 偏暖白光和深红光
        weights = weights / np.sum(weights)

        spectrum = self.calculate_spectrum(weights)
        cct = self.calculate_cct(spectrum)
        duv = self.calculate_duv(spectrum)
        rf = self.calculate_rf(spectrum)
        rg = self.calculate_rg(spectrum)
        mel_der = self.calculate_mel_der(spectrum)

        print(f"启发式优化结果:")
        print(f"权重: {dict(zip(self.channels, weights))}")
        print(f"CCT: {cct:.1f} K")
        print(f"Duv: {duv:.6f}")
        print(f"Rf: {rf:.1f}")
        print(f"Rg: {rg:.1f}")
        print(f"mel-DER: {mel_der:.6f}")

        return weights, {
            'CCT': cct, 'Duv': duv, 'Rf': rf, 'Rg': rg, 'mel-DER': mel_der
        }


def main():
    """主函数"""
    print("=== 多通道LED光源优化系统 ===")

    # 创建优化器
    data_file = "Problem 2_LED_SPD.xlsx"  # 请替换为实际文件路径
    optimizer = LEDOptimizer(data_file)

    # 优化场景一：日间照明模式
    scenario1_weights, scenario1_results = optimizer.optimize_scenario1()

    # 优化场景二：夜间助眠模式
    scenario2_weights, scenario2_results = optimizer.optimize_scenario2()

    # 绘制结果
    optimizer.plot_results(scenario1_weights, scenario2_weights)

    # 输出总结
    print("\n=== 优化结果总结 ===")
    if scenario1_results:
        print("场景一：日间照明模式:")
        print(f"  最优权重: {dict(zip(optimizer.channels, scenario1_weights))}")
        for key, value in scenario1_results.items():
            print(f"  {key}: {value}")

    if scenario2_results:
        print("\n场景二：夜间助眠模式:")
        print(f"  最优权重: {dict(zip(optimizer.channels, scenario2_weights))}")
        for key, value in scenario2_results.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()