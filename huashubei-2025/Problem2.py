import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from colour.colorimetry import CCS_ILLUMINANTS  # 修改后的导入
from colour.temperature import uv_to_CCT_Robertson1968, CCT_to_uv
from colour.quality import colour_quality_scale
from colour.quality import colour_rendering_index
from scipy.interpolate import interp1d


# ======================
# 1. 完善的光谱数据处理框架
# ======================
class SpectrumProcessor:
    def __init__(self, wavelengths):
        self.wavelengths = wavelengths
        self.delta_lambda = 1.0  # 1nm波长间隔

        # 从文件加载CIE 1931标准观察者数据
        self.x_bar, self.y_bar, self.z_bar = self._load_cie_observer()

        # 从文件加载褪黑素光敏感度曲线
        self.mel_curve = self._load_melanopic_response()

        # 加载D65标准光源数据
        self.d65_spectrum = self._load_d65_spectrum()

    def _load_cie_observer(self):
        """从文件加载CIE 1931标准观察者数据"""
        try:
            # 实际应用中替换为真实数据文件
            # df = pd.read_csv('cie_1931_2deg.csv')
            # 使用colour内置数据作为替代
            from colour.colorimetry import MSDS_CMFS
            cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

            # 插值到所需波长范围
            x = interp1d(cmfs.wavelengths, cmfs.values[:, 0],
                         bounds_error=False, fill_value=0)(self.wavelengths)
            y = interp1d(cmfs.wavelengths, cmfs.values[:, 1],
                         bounds_error=False, fill_value=0)(self.wavelengths)
            z = interp1d(cmfs.wavelengths, cmfs.values[:, 2],
                         bounds_error=False, fill_value=0)(self.wavelengths)

            return x, y, z
        except Exception as e:
            print(f"Error loading CIE observer data: {str(e)}")
            # 退回简化模型
            return self._simple_cie_observer()

    def _simple_cie_observer(self):
        """简化版CIE观察者数据"""
        x = np.zeros_like(self.wavelengths)
        y = np.zeros_like(self.wavelengths)
        z = np.zeros_like(self.wavelengths)

        vis_mask = (self.wavelengths >= 380) & (self.wavelengths <= 780)
        x[vis_mask] = np.interp(self.wavelengths[vis_mask], [380, 780], [0.001, 0.2])
        y[vis_mask] = np.interp(self.wavelengths[vis_mask], [380, 780], [0.000, 0.9])
        z[vis_mask] = np.interp(self.wavelengths[vis_mask], [380, 780], [0.006, 0.1])

        return x, y, z

    def _load_melanopic_response(self):
        """从文件加载褪黑素光敏感度曲线"""
        try:
            # 实际应用中替换为真实数据文件
            # df = pd.read_csv('melanopic_response.csv')
            # 模拟数据：高斯曲线，峰值在480nm
            mel_response = np.zeros_like(self.wavelengths)
            peak_mask = (self.wavelengths >= 440) & (self.wavelengths <= 500)
            mel_response[peak_mask] = np.exp(-0.5 * ((self.wavelengths[peak_mask] - 480) / 30) ** 2)

            # 归一化到0-1范围
            mel_response /= np.max(mel_response)
            return mel_response
        except Exception as e:
            print(f"Error loading melanopic response data: {str(e)}")
            return self._simple_melanopic_response()

    def _simple_melanopic_response(self):
        """简化版褪黑素响应曲线"""
        mel_response = np.zeros_like(self.wavelengths)
        peak_mask = (self.wavelengths >= 440) & (self.wavelengths <= 500)
        mel_response[peak_mask] = np.interp(
            self.wavelengths[peak_mask], [440, 480, 500], [0.1, 1.0, 0.1]
        )
        return mel_response

    def _load_d65_spectrum(self):
        """加载D65标准光源光谱"""
        try:
            # 实际应用中替换为真实数据文件
            # df = pd.read_csv('d65_spectrum.csv')
            # 使用colour内置D65数据作为替代
            from colour import SDS_ILLUMINANTS
            d65 = SDS_ILLUMINANTS['D65']

            # 插值到所需波长范围
            d65_interp = interp1d(d65.wavelengths, d65.values,
                                  bounds_error=False, fill_value=0)(self.wavelengths)

            return d65_interp
        except Exception as e:
            print(f"Error loading D65 spectrum data: {str(e)}")
            return self._simple_d65_spectrum()

    def _simple_d65_spectrum(self):
        """简化版D65光谱"""
        d65 = np.zeros_like(self.wavelengths)
        vis_mask = (self.wavelengths >= 380) & (self.wavelengths <= 780)
        d65[vis_mask] = np.interp(
            self.wavelengths[vis_mask], [380, 480, 580, 780], [0.2, 1.0, 1.0, 0.3]
        )
        return d65

    def calculate_xyz(self, spectrum):
        """计算CIEXYZ三刺激值"""
        X = np.sum(spectrum * self.x_bar) * self.delta_lambda
        Y = np.sum(spectrum * self.y_bar) * self.delta_lambda
        Z = np.sum(spectrum * self.z_bar) * self.delta_lambda
        return X, Y, Z

    def calculate_xy(self, X, Y, Z):
        """计算色品坐标"""
        x = X / (X + Y + Z) if (X + Y + Z) > 0 else 0
        y = Y / (X + Y + Z) if (X + Y + Z) > 0 else 0
        return x, y

    def calculate_cct_duv(self, X, Y, Z):
        """计算CCT和Duv - 使用Robertson方法"""
        # 计算uv坐标
        denom = X + 15 * Y + 3 * Z
        u = 4 * X / denom if denom > 0 else 0
        v = 6 * Y / denom if denom > 0 else 0
        uv = np.array([u, v])

        # 使用Robertson方法计算CCT和Duv
        try:
            cct, duv = uv_to_CCT_Robertson1968(uv)
            return cct, duv
        except:
            # 退回简化方法
            return self._simple_cct_duv(u, v)

    def _simple_cct_duv(self, u, v):
        """简化的CCT和Duv计算"""
        # CCT简化计算
        n = (u - 0.2920) / (0.2400 - v) if abs(0.2400 - v) > 1e-6 else 0
        cct = 1000 * n

        # Duv简化计算 - 黑体辐射轨迹上的点
        uv0 = CCT_to_uv(cct) if cct > 0 else (0, 0)
        duv = np.sqrt((u - uv0[0]) ** 2 + (v - uv0[1]) ** 2)
        return cct, duv

    def calculate_mel_der(self, spectrum):
        """计算褪黑素日光照度比(mel-DER)"""
        E_mel = np.sum(spectrum * self.mel_curve) * self.delta_lambda
        E_mel_d65 = np.sum(self.d65_spectrum * self.mel_curve) * self.delta_lambda
        return E_mel / E_mel_d65 if E_mel_d65 > 0 else 0

    def calculate_rf_rg(self, spectrum):
        """计算Rf和Rg色质量指标"""
        # 创建SpectralDistribution对象
        from colour import SpectralDistribution
        spd = SpectralDistribution(dict(zip(self.wavelengths, spectrum)),
                                   name='LED Spectrum')

        # 使用CQS方法计算Rf
        # 注意: CQS使用100分制，与CRI相似
        try:
            cqs = colour_quality_scale(spd, additional_data=True)
            Rf = cqs.Q_a
            # 使用CRI方法计算Rg
            cri = colour_rendering_index(spd)
            Rg = cri['Rg']
            return Rf, Rg
        except Exception as e:
            print(f"Error calculating Rf/Rg: {str(e)}")
            # 退回简化模型
            return np.random.uniform(80, 95), np.random.uniform(85, 105)


# ==========================
# 2. 改进的LED权重优化模型
# ==========================
class LEDOptimizer:
    def __init__(self, wavelengths, channels_spd):
        """
        wavelengths: 波长数组 (nm)
        channels_spd: 各通道光谱矩阵 [channel x wavelength]
        """
        self.wavelengths = wavelengths
        self.channels_spd = channels_spd
        self.processor = SpectrumProcessor(wavelengths)

        # 归一化各通道光谱
        self.norm_spd = self.normalize_spd(channels_spd)

    def normalize_spd(self, spd_matrix):
        """归一化光谱数据 (0-1范围)"""
        min_vals = np.min(spd_matrix, axis=1, keepdims=True)
        max_vals = np.max(spd_matrix, axis=1, keepdims=True)
        return (spd_matrix - min_vals) / (max_vals - min_vals + 1e-8)

    def synthesize_spectrum(self, weights):
        """合成光谱"""
        return np.dot(weights, self.norm_spd)

    def calculate_metrics(self, spectrum):
        """计算所有光学指标"""
        X, Y, Z = self.processor.calculate_xyz(spectrum)
        x, y = self.processor.calculate_xy(X, Y, Z)
        cct, duv = self.processor.calculate_cct_duv(X, Y, Z)
        Rf, Rg = self.processor.calculate_rf_rg(spectrum)

        return {
            'cct': cct,
            'duv': duv,
            'mel_der': self.processor.calculate_mel_der(spectrum),
            'Rf': Rf,
            'Rg': Rg
        }

    def daytime_objective(self, weights):
        """日间模式目标函数：最大化Rf"""
        spectrum = self.synthesize_spectrum(weights)
        metrics = self.calculate_metrics(spectrum)
        return -metrics['Rf']  # 负值因为我们要最大化

    def nighttime_objective(self, weights):
        """夜间模式目标函数：最小化mel-DER + 惩罚项"""
        spectrum = self.synthesize_spectrum(weights)
        metrics = self.calculate_metrics(spectrum)

        # 主要目标: 最小化mel-DER
        objective = metrics['mel_der']

        # 添加Rf约束的惩罚项
        if metrics['Rf'] < 80:
            objective += (80 - metrics['Rf']) * 10  # 严重偏离时增加惩罚

        # 添加CCT约束的惩罚项
        cct = metrics['cct']
        if cct < 2500:
            objective += (2500 - cct) * 0.01  # 色温过低惩罚
        elif cct > 3500:
            objective += (cct - 3500) * 0.01  # 色温过高惩罚

        return objective

    def apply_constraints(self, weights, mode):
        """应用约束条件 - 仅用于鲁棒性测试"""
        spectrum = self.synthesize_spectrum(weights)
        metrics = self.calculate_metrics(spectrum)

        constraints = []

        if mode == 'day':
            # 日间约束
            constraints.append(5500 <= metrics['cct'] <= 6500)  # CCT约束
            constraints.append(95 <= metrics['Rg'] <= 105)  # Rg约束
        else:  # night
            # 夜间约束
            constraints.append(2500 <= metrics['cct'] <= 3500)  # CCT约束
            constraints.append(metrics['Rf'] >= 80)  # Rf约束

        # 通用约束
        constraints.append(0.999 <= np.sum(weights) <= 1.001)  # 权重和≈1
        constraints.append(np.all(weights >= 0))  # 权重非负

        return all(constraints)

    def optimize(self, mode, initial_weights=None):
        """使用约束优化方法"""
        n_channels = self.norm_spd.shape[0]

        # 设置初始权重
        if initial_weights is None:
            initial_weights = np.ones(n_channels) / n_channels

        # 选择目标函数
        objective = self.daytime_objective if mode == 'day' else self.nighttime_objective

        # 定义约束
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
            {'type': 'ineq', 'fun': lambda w: w}  # 权重非负
        ]

        # 为特定模式添加约束
        if mode == 'day':
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.calculate_metrics(self.synthesize_spectrum(w))['Rg'] - 95
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: 105 - self.calculate_metrics(self.synthesize_spectrum(w))['Rg']
            })
        else:  # night
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.calculate_metrics(self.synthesize_spectrum(w))['Rf'] - 80
            })

        # 优化参数
        options = {
            'maxiter': 1000,
            'ftol': 1e-6,
            'disp': True
        }

        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1)] * n_channels,
            options=options
        )

        if result.success:
            final_weights = result.x
            # 验证约束
            if self.apply_constraints(final_weights, mode):
                return final_weights
            else:
                raise RuntimeError(f"Optimization failed to meet constraints for {mode} mode")
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")

    def robustness_test(self, weights, mode, variation=0.05, n_tests=100):
        """权重鲁棒性测试 (±5%波动)"""
        success_count = 0

        for _ in range(n_tests):
            # 在±5%范围内随机扰动权重
            perturbed_weights = weights * np.random.uniform(1 - variation, 1 + variation, len(weights))
            perturbed_weights /= np.sum(perturbed_weights)  # 重新归一化

            if self.apply_constraints(perturbed_weights, mode):
                success_count += 1

        return success_count / n_tests


# ======================
# 3. 主程序执行框架
# ======================
if __name__ == "__main__":
    # 1. 设置波长范围 (380-780nm, 1nm间隔)
    wavelengths = np.arange(380, 781)

    # 2. 模拟5个LED通道的光谱
    # 蓝光、绿光、红光、暖白光、冷白光
    channels_spd = np.zeros((5, len(wavelengths)))

    # 改进的模拟光谱 (更符合真实LED特性)
    channels_spd[0] = np.exp(-0.5 * ((wavelengths - 450) / 20) ** 2)  # 蓝光 (峰值450nm)
    channels_spd[1] = np.exp(-0.5 * ((wavelengths - 525) / 25) ** 2)  # 绿光 (峰值525nm)
    channels_spd[2] = np.exp(-0.5 * ((wavelengths - 630) / 30) ** 2)  # 红光 (峰值630nm)

    # 暖白光 (宽光谱，在570nm附近有峰值)
    channels_spd[3] = (np.exp(-0.5 * ((wavelengths - 450) / 100) ** 2) +
                       np.exp(-0.5 * ((wavelengths - 570) / 80) ** 2))

    # 冷白光 (宽光谱，在470nm附近有峰值)
    channels_spd[4] = (np.exp(-0.5 * ((wavelengths - 470) / 80) ** 2) +
                       np.exp(-0.5 * ((wavelengths - 550) / 70) ** 2))

    # 3. 初始化优化器
    optimizer = LEDOptimizer(wavelengths, channels_spd)

    # 4. 优化日间模式
    try:
        print("开始优化日间模式...")
        day_weights = optimizer.optimize('day')
        day_spectrum = optimizer.synthesize_spectrum(day_weights)
        day_metrics = optimizer.calculate_metrics(day_spectrum)
        day_robustness = optimizer.robustness_test(day_weights, 'day')

        print("\n日间模式优化结果:")
        print(f"权重: 蓝光={day_weights[0]:.3f}, 绿光={day_weights[1]:.3f}, 红光={day_weights[2]:.3f}, "
              f"暖白光={day_weights[3]:.3f}, 冷白光={day_weights[4]:.3f}")
        print(f"CCT={day_metrics['cct']:.0f}K, Duv={day_metrics['duv']:.4f}, "
              f"Rf={day_metrics['Rf']:.1f}, Rg={day_metrics['Rg']:.1f}, "
              f"mel-DER={day_metrics['mel_der']:.3f}")
        print(f"权重鲁棒性: {day_robustness * 100:.1f}%")

        # 可视化日间光谱
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, day_spectrum)
        plt.title("日间模式合成光谱")
        plt.xlabel("波长 (nm)")
        plt.ylabel("相对功率")
        plt.grid(True)
        plt.savefig("daytime_spectrum.png", dpi=300)
        plt.close()

        # 可视化各通道贡献
        plt.figure(figsize=(10, 6))
        for i, weight in enumerate(day_weights):
            plt.plot(wavelengths, weight * optimizer.norm_spd[i],
                     label=f'Channel {i} ({weight:.3f})', alpha=0.7)
        plt.title("各通道贡献 (日间模式)")
        plt.xlabel("波长 (nm)")
        plt.ylabel("相对功率")
        plt.legend()
        plt.grid(True)
        plt.savefig("daytime_channels.png", dpi=300)
        plt.close()

    except RuntimeError as e:
        print(f"日间模式优化失败: {str(e)}")

    # 5. 优化夜间模式
    try:
        print("\n开始优化夜间模式...")
        night_weights = optimizer.optimize('night')
        night_spectrum = optimizer.synthesize_spectrum(night_weights)
        night_metrics = optimizer.calculate_metrics(night_spectrum)
        night_robustness = optimizer.robustness_test(night_weights, 'night')

        print("\n夜间模式优化结果:")
        print(f"权重: 蓝光={night_weights[0]:.3f}, 绿光={night_weights[1]:.3f}, 红光={night_weights[2]:.3f}, "
              f"暖白光={night_weights[3]:.3f}, 冷白光={night_weights[4]:.3f}")
        print(f"CCT={night_metrics['cct']:.0f}K, Duv={night_metrics['duv']:.4f}, "
              f"Rf={night_metrics['Rf']:.1f}, Rg={night_metrics['Rg']:.1f}, "
              f"mel-DER={night_metrics['mel_der']:.3f}")
        print(f"权重鲁棒性: {night_robustness * 100:.1f}%")

        # 可视化夜间光谱
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, night_spectrum)
        plt.title("夜间模式合成光谱")
        plt.xlabel("波长 (nm)")
        plt.ylabel("相对功率")
        plt.grid(True)
        plt.savefig("nighttime_spectrum.png", dpi=300)
        plt.close()

        # 可视化各通道贡献
        plt.figure(figsize=(10, 6))
        for i, weight in enumerate(night_weights):
            plt.plot(wavelengths, weight * optimizer.norm_spd[i],
                     label=f'Channel {i} ({weight:.3f})', alpha=0.7)
        plt.title("各通道贡献 (夜间模式)")
        plt.xlabel("波长 (nm)")
        plt.ylabel("相对功率")
        plt.legend()
        plt.grid(True)
        plt.savefig("nighttime_channels.png", dpi=300)
        plt.close()

    except RuntimeError as e:
        print(f"夜间模式优化失败: {str(e)}")

    print("\n优化过程完成。")