import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
from datetime import datetime, time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ========== 1. 读取数据 ==========
sun_df = pd.read_csv('./data/Problem3.csv', encoding='gbk')
led_df = pd.read_csv('data/Problem2.csv', encoding='gbk')

# 提取波长和LED通道数据
wavelengths = np.arange(380, 781)
led_spds = led_df.iloc[:, 1:].values.T  # shape: (5, 401)
led_channels = ['深红光', '绿光', '蓝光', '暖白光', '冷白光']

# 处理太阳光谱数据 - 只保留8:30到19:30的时间点
sun_df.columns.values[0] = '波长'
sun_df['波长'] = sun_df['波长'].str.extract(r'(\d+)').astype(int)
sun_df = sun_df.set_index('波长').loc[380:780]

# 只保留8:30到19:30的时间点
all_time_points = sun_df.columns.tolist()
start_index = all_time_points.index('8:30')
end_index = all_time_points.index('19:30') + 1
time_points = all_time_points[start_index:end_index]
sun_spectra = sun_df.iloc[:, start_index:end_index].values.T  # shape: (12, 401)

print(f"优化时间点: {time_points}")


# ========== 2. 光谱合成函数 ==========
def synthesize_spd(weights):
    return np.dot(weights, led_spds)  # shape: (401,)


# ========== 3. 色温计算（McCamy近似法） ==========
def compute_cct(spd):
    # 色匹配函数数据
    cmf_x = np.loadtxt('./data/cmf_x.txt')
    cmf_y = np.loadtxt('./data/cmf_y.txt')
    cmf_z = np.loadtxt('./data/cmf_z.txt')

    # 提取对应波长380-780nm的数据
    cmf_380_780 = pd.DataFrame({
        'wl': cmf_x[:, 0],
        'x': cmf_x[:, 1],
        'y': cmf_y[:, 1],
        'z': cmf_z[:, 1]
    }).set_index('wl').loc[380:780].reset_index()

    # 三刺激值计算
    X = np.sum(spd * cmf_380_780['x'])
    Y = np.sum(spd * cmf_380_780['y'])
    Z = np.sum(spd * cmf_380_780['z'])

    # 色坐标
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    # McCamy公式
    n = (x - 0.3320) / (0.1858 - y)
    CCT = 437 * n ** 3 + 3601 * n ** 2 + 6861 * n + 5517
    return CCT


# ========== 4. mel-DER 指标计算 ==========
def compute_mel_der(spd):
    mel_curve = np.zeros(len(wavelengths))
    mel_curve[(wavelengths >= 460) & (wavelengths <= 480)] = 1
    return np.sum(spd * mel_curve) / np.sum(spd)


# ========== 5. 计算太阳光谱指标 ==========
sun_cct = []
sun_mel_der = []

for spd in sun_spectra:
    sun_cct.append(compute_cct(spd))
    sun_mel_der.append(compute_mel_der(spd))


# ========== 6. 优化函数 ==========
def fit_single_time(target_spd, target_mel_der):
    def error_func(weights):
        synth_spd = synthesize_spd(weights)

        # 1. 计算光谱余弦相似度
        dot_product = np.dot(synth_spd, target_spd)
        norm_synth = np.linalg.norm(synth_spd)
        norm_target = np.linalg.norm(target_spd)
        cosine_sim = dot_product / (norm_synth * norm_target) if norm_synth > 0 and norm_target > 0 else 0

        # 2. 计算mel-DER绝对误差
        synth_mel_der = compute_mel_der(synth_spd)
        mel_error = np.abs(synth_mel_der - target_mel_der)

        # 3. 组合目标函数（α=0.6）
        return 0.6 * (1 - cosine_sim) + 0.4 * mel_error

    # 约束条件（权重归一化和非负）
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w})

    # 初始权重（均匀分布）
    result = minimize(error_func, x0=np.ones(5) / 5,
                      method='SLSQP', bounds=[(0, 1)] * 5,
                      constraints=cons,
                      options={'maxiter': 1000, 'ftol': 1e-6})

    return result.x if result.success else np.ones(5) / 5


# ========== 7. 优化所有时间点 ==========
weights_over_time = []
synth_cct = []
synth_mel_der = []

print("=" * 120)
print(
    "时间点 |   权重分布 [深红光, 绿光, 蓝光, 暖白光, 冷白光]   | 合成CCT(K) | 太阳CCT(K) | 合成mel-DER | 太阳mel-DER | CCT误差 | mel-DER误差 | 光谱相似度")
print("-" * 120)

for i, time in enumerate(time_points):
    target_spd = sun_spectra[i]
    target_mel_der = sun_mel_der[i]

    # 优化权重
    weights = fit_single_time(target_spd, target_mel_der)
    synth_spd = synthesize_spd(weights)

    # 计算指标
    cct_val = compute_cct(synth_spd)
    mel_der_val = compute_mel_der(synth_spd)

    # 计算光谱相似度
    dot_product = np.dot(synth_spd, target_spd)
    norm_synth = np.linalg.norm(synth_spd)
    norm_target = np.linalg.norm(target_spd)
    cosine_sim = dot_product / (norm_synth * norm_target) if norm_synth > 0 and norm_target > 0 else 0

    # 保存结果
    weights_over_time.append(weights)
    synth_cct.append(cct_val)
    synth_mel_der.append(mel_der_val)

    # 格式化输出
    weight_str = '[' + ', '.join([f'{w:.3f}' for w in weights]) + ']'
    cct_error = cct_val - sun_cct[i]
    mel_error = mel_der_val - target_mel_der

    print(
        f"{time:^7}| {weight_str} | {int(cct_val):>9} | {int(sun_cct[i]):>9} | {mel_der_val:>10.4f} | {target_mel_der:>10.4f} | {cct_error:>+7.0f} | {mel_error:>+10.4f} | {cosine_sim:.4f}")

print("=" * 120)

weights_over_time = np.array(weights_over_time)

# ========== 8. 时间插值 ==========
# 将时间转换为分钟（从8:30开始）
time_minutes = [(int(time.split(':')[0]) - 8) * 60 + int(time.split(':')[1]) - 30 for time in time_points]

# 创建插值函数
interp_functions = []
for i in range(5):
    interp_functions.append(
        interp1d(time_minutes, weights_over_time[:, i], kind='linear', bounds_error=False, fill_value="extrapolate"))

# 生成全天连续时间点（每5分钟一个点）
full_time_minutes = np.arange(0, max(time_minutes) + 5, 5)
full_time_str = [f"{8 + (t + 30) // 60}:{(t + 30) % 60:02d}" for t in full_time_minutes]

# 插值得到全天权重
full_weights = np.zeros((len(full_time_minutes), 5))
for i in range(5):
    full_weights[:, i] = interp_functions[i](full_time_minutes)

# ========== 9. 可视化 ==========
# 9.1 权重分布图
plt.figure(figsize=(14, 8))
colors = ['#8B0000', 'green', 'blue', '#FFA500', '#ADD8E6']  # 深红,绿,蓝,橙(暖白),浅蓝(冷白)
for i in range(5):
    plt.plot(full_time_str, full_weights[:, i],
             '-', markersize=4, label=led_channels[i], color=colors[i], linewidth=2.5)

plt.title('LED通道权重全天动态变化 (8:30-19:30)', fontsize=16)
plt.xlabel('时间')
plt.ylabel('权重')
plt.legend(title='LED通道', loc='upper right')
plt.xticks(time_points, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 0.7)
plt.tight_layout()
plt.savefig('./figures/权重变化.jpg', dpi=300)
plt.show()

# 9.2 色温对比图
full_synth_cct = []
full_synth_mel = []

for weights in full_weights:
    spd = synthesize_spd(weights)
    full_synth_cct.append(compute_cct(spd))
    full_synth_mel.append(compute_mel_der(spd))

# 插值获取全天的太阳指标
sun_cct_interp = interp1d(time_minutes, sun_cct, kind='linear', bounds_error=False, fill_value="extrapolate")
sun_mel_interp = interp1d(time_minutes, sun_mel_der, kind='linear', bounds_error=False, fill_value="extrapolate")
full_sun_cct = sun_cct_interp(full_time_minutes)
full_sun_mel = sun_mel_interp(full_time_minutes)

plt.figure(figsize=(14, 8))
plt.plot(full_time_str, full_synth_cct, 'r-', label='合成光谱色温', linewidth=2.5)
plt.plot(full_time_str, full_sun_cct, 'b--', label='太阳光谱色温', linewidth=2.5)
plt.title('色温全天对比 (8:30-19:30)', fontsize=16)
plt.xlabel('时间')
plt.ylabel('色温 (K)')
plt.legend()
plt.xticks(time_points, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./figures/色温对比.jpg', dpi=300)
plt.show()

# 9.3 mel-DER对比图
plt.figure(figsize=(14, 8))
plt.plot(full_time_str, full_synth_mel, 'r-', label='合成光谱mel-DER', linewidth=2.5)
plt.plot(full_time_str, full_sun_mel, 'b--', label='太阳光谱mel-DER', linewidth=2.5)
plt.title('mel-DER全天对比 (8:30-19:30)', fontsize=16)
plt.xlabel('时间')
plt.ylabel('mel-DER')
plt.legend()
plt.xticks(time_points, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./figures/melDER对比.jpg', dpi=300)
plt.show()


# ========== 10. 代表性时间点光谱对比 ==========
def plot_spectrum_comparison(time_str, index):
    target_spd = sun_spectra[index]
    weights = weights_over_time[index]
    synth_spd = synthesize_spd(weights)

    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, target_spd, label=f'太阳光谱 ({time_str})', linewidth=2.5, color='darkorange')
    plt.plot(wavelengths, synth_spd, '--', label='LED合成光谱', linewidth=2.5, color='royalblue')

    # 添加mel响应区域
    plt.fill_between(wavelengths, 0, max(target_spd) * 1.1,
                     where=(wavelengths >= 460) & (wavelengths <= 480),
                     color='blue', alpha=0.2, label='mel响应区 (460-480nm)')

    # 计算光谱相似度
    dot_product = np.dot(synth_spd, target_spd)
    norm_synth = np.linalg.norm(synth_spd)
    norm_target = np.linalg.norm(target_spd)
    cosine_sim = dot_product / (norm_synth * norm_target) if norm_synth > 0 and norm_target > 0 else 0

    plt.title(f'光谱对比 - {time_str} (相似度: {cosine_sim:.4f})', fontsize=16)
    plt.xlabel('波长 (nm)')
    plt.ylabel('相对辐射强度')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'./figures/光谱对比_{time_str.replace(":", "")}.jpg', dpi=300)
    plt.show()

    # 打印详细指标
    print(f"\n{time_str} 详细指标:")
    print(f"  合成光谱CCT: {compute_cct(synth_spd):.0f}K, 太阳光谱CCT: {sun_cct[index]:.0f}K")
    print(f"  合成光谱mel-DER: {compute_mel_der(synth_spd):.4f}, 太阳光谱mel-DER: {sun_mel_der[index]:.4f}")
    print(f"  权重分布: {dict(zip(led_channels, weights.round(3)))}")


# 选取三个代表性时间点：早晨(8:30)、正午(12:30)、傍晚(18:30)
plot_spectrum_comparison("8:30", 0)
plot_spectrum_comparison("12:30", 4)  # 12:30是第5个时间点（索引4）
plot_spectrum_comparison("18:30", 10)  # 18:30是第11个时间点（索引10）

print("\n动态节律模拟完成! 所有图表已保存至 ./figures/ 目录")