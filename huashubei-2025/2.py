import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# --------------------------
# 步骤1：数据输入（五通道SPD特性）
# 通道顺序：深红光、绿光、蓝光、暖白光、冷白光
# --------------------------
channels = ["深红光", "绿光", "蓝光", "暖白光", "冷白光"]
np.random.seed(42)  # 固定随机数，保证结果可复现

# 各通道光谱特性（对色温/生理效应的影响权重）
effect_cct = np.array([-500, 0, 1500, -1000, 2000])  # 影响色温（CCT）
effect_mel = np.array([0.1, 0.2, 0.8, 0.3, 0.6])     # 影响mel-DER（生理节律）
effect_rf = np.array([0.7, 0.8, 0.6, 0.9, 0.85])     # 影响Rf（显色指数）

# --------------------------
# 步骤2：优化函数（分场景定义目标与约束）
# --------------------------
def optimize_weights(is_daytime):
    if is_daytime:
        # 日间模式：最大化Rf（linprog默认最小化，故取负系数）
        c = -effect_rf
        # 约束1：CCT∈5500-6500K（中性白光）
        A_ub = [effect_cct, -effect_cct]
        b_ub = [6500, -5500]  # 6500≥CCT≥5500
        # 约束2：暖白光+冷白光权重≥0.3（简化保证Rg在合理范围）
        A_ub.append([0, 0, 0, -1, -1])
        b_ub.append(-0.3)
        # 约束3：Rf≥88（保证显色性）
        A_ub.append(-effect_rf)
        b_ub.append(-88)
    else:
        # 夜间模式：最小化mel-DER（减少生理节律干扰）
        c = effect_mel
        # 约束1：CCT∈2500-3500K（暖黄光）
        A_ub = [effect_cct, -effect_cct]
        b_ub = [3500, -2500]  # 3500≥CCT≥2500
        # 约束2：Rf≥80（基础显色性）
        A_ub.append(-effect_rf)
        b_ub.append(-80)
    
    # 通用约束：权重和为1，且各权重∈[0,1]
    A_eq = [[1, 1, 1, 1, 1]]
    b_eq = [1]
    bounds = [(0, 1)] * 5  # 权重范围
    
    # 求解线性规划
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    # 检查求解是否成功（新增：避免无可行解时出错）
    if not result.success:
        raise ValueError(f"求解失败：{result.message}（请检查约束条件是否冲突）")
    
    weights = result.x  # 最优权重
    
    # 计算合成参数
    cct = np.sum(weights * effect_cct)
    rf = np.sum(weights * effect_rf)
    mel_der = np.sum(weights * effect_mel)
    # 简化计算Rg（仅日间模式）
    rg = 95 + np.sum(weights[:2]) * 10 if is_daytime else None
    
    return weights, cct, rf, rg, mel_der

# --------------------------
# 步骤3：求解并处理结果
# --------------------------
try:
    # 求解日间和夜间模式
    day_weights, day_cct, day_rf, day_rg, day_mel = optimize_weights(is_daytime=True)
    night_weights, night_cct, night_rf, _, night_mel = optimize_weights(is_daytime=False)

    # 格式化输出（明确通道与权重对应关系）
    print("=== 日间照明模式优化结果 ===")
    print("通道权重：")
    for chan, w in zip(channels, day_weights):
        print(f"  {chan}：{w:.3f}")
    print(f"\n合成参数：")
    print(f"  色温（CCT）：{day_cct:.0f}K")
    print(f"  显色指数（Rf）：{day_rf:.1f}")
    print(f"  色域指数（Rg）：{day_rg:.1f}")
    print(f"  生理节律影响（mel-DER）：{day_mel:.3f}")

    print("\n=== 夜间助眠模式优化结果 ===")
    print("通道权重：")
    for chan, w in zip(channels, night_weights):
        print(f"  {chan}：{w:.3f}")
    print(f"\n合成参数：")
    print(f"  色温（CCT）：{night_cct:.0f}K")
    print(f"  显色指数（Rf）：{night_rf:.1f}")
    print(f"  生理节律影响（mel-DER）：{night_mel:.3f}")

    # --------------------------
    # 步骤4：可视化优化（颜色更贴合光色）
    # --------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#8B0000', '#006400', '#0000CD', '#FFA500', '#87CEEB']  # 深红光/绿光/蓝光/暖黄/浅蓝

    # 日间模式权重图
    ax1.bar(channels, day_weights, color=colors)
    ax1.set_title("日间模式LED通道权重分布", fontsize=12)
    ax1.set_xlabel("LED通道", fontsize=10)
    ax1.set_ylabel("权重值", fontsize=10)
    ax1.set_ylim(0, 0.8)
    ax1.grid(axis='y', alpha=0.3)
    # 标注权重值
    for i, w in enumerate(day_weights):
        ax1.text(i, w + 0.02, f"{w:.3f}", ha='center', fontsize=9)

    # 夜间模式权重图
    ax2.bar(channels, night_weights, color=colors)
    ax2.set_title("夜间模式LED通道权重分布", fontsize=12)
    ax2.set_xlabel("LED通道", fontsize=10)
    ax2.set_ylabel("权重值", fontsize=10)
    ax2.set_ylim(0, 0.8)
    ax2.grid(axis='y', alpha=0.3)
    # 标注权重值
    for i, w in enumerate(night_weights):
        ax2.text(i, w + 0.02, f"{w:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"运行错误：{e}")