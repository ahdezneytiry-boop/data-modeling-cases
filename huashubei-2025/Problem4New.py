import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_rel, wilcoxon, friedmanchisquare
import warnings

warnings.filterwarnings('ignore')  # 忽略正态性检验可能出现的警告

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)  # 设置默认图形大小
plt.rcParams['figure.dpi'] = 100  # 设置默认分辨率

# ======================
# 1. 加载与预处理数据
# ======================

# 加载原始数据
raw_data = pd.read_csv('./data/Problem4.csv', encoding='gbk', header=None)

# 提取每个被试的三晚数据（环境A, B, C），前两行分别是被试标签和夜晚标签
subjects = [f'被试{i + 1}' for i in range(11)]
env_labels = ['A', 'B', 'C']  # Night1, Night2, Night3 -> A, B, C
records_per_subject = {}

# 解析每位被试的三晚数据
for i, subj in enumerate(subjects):
    cols = raw_data.iloc[0:2, i * 3:(i + 1) * 3].values.tolist()
    nights = raw_data.iloc[2:, i * 3:(i + 1) * 3].astype('float')
    nights.columns = env_labels
    records_per_subject[subj] = nights.dropna(how='all').reset_index(drop=True)


# ======================
# 2. 指标计算函数定义
# ======================

def compute_metrics(stage_seq):
    """
    根据30秒间隔的睡眠阶段序列，计算各类睡眠指标
    """
    stage_seq = np.array(stage_seq)
    T_total = len(stage_seq) * 30  # 总卧床时间（秒）

    is_sleep = np.isin(stage_seq, [2, 3, 5])
    is_deep = (stage_seq == 3)
    is_rem = (stage_seq == 5)

    # 总睡眠时间
    TST = np.sum(is_sleep) * 30

    # 睡眠效率
    SE = (TST / T_total) * 100 if T_total else np.nan

    # 入睡潜伏期
    sleep_indices = np.where(is_sleep)[0]
    SOL = sleep_indices[0] * 30 if len(sleep_indices) else np.nan

    # 深睡比例
    N3 = np.sum(is_deep) * 30
    N3_percent = (N3 / TST) * 100 if TST else np.nan

    # REM比例
    REM = np.sum(is_rem) * 30
    REM_percent = (REM / TST) * 100 if TST else np.nan

    # 夜间清醒次数（睡后出现的清醒次数）
    if len(sleep_indices) == 0:
        awakenings = np.nan
    else:
        first_sleep = sleep_indices[0]
        post_sleep = stage_seq[first_sleep:]
        wake_seq = (post_sleep == 4).astype(int)
        transitions = np.diff(wake_seq)
        awakenings = np.sum(transitions == 1)

    return {
        'TST': TST / 60,  # 转为分钟
        'SE': SE,
        'SOL': SOL / 60 if SOL else np.nan,  # 分钟
        'N3%': N3_percent,
        'REM%': REM_percent,
        'NoA': awakenings
    }


# ======================
# 3. 计算每个被试的指标
# ======================

metrics_all = []

for subj, df in records_per_subject.items():
    for env in env_labels:
        stages = df[env].dropna().astype(int).values
        metrics = compute_metrics(stages)
        metrics.update({'Subject': subj, 'Environment': env})
        metrics_all.append(metrics)

metrics_df = pd.DataFrame(metrics_all)

# 转换格式用于绘图与统计
metrics_long = metrics_df.melt(id_vars=['Subject', 'Environment'],
                               var_name='Metric', value_name='Value')

# ======================
# 4. 统计检验部分
# ======================

results = []
print("\n" + "=" * 80)
print("睡眠指标统计检验结果")
print("=" * 80)
print("说明:")
print("1. Normality: 各环境数据是否满足正态分布 (True/False)")
print("2. Friedman_p: Friedman检验整体显著性")
print("3. p(A vs B): A与B环境配对检验p值")
print("4. p(A vs C): A与C环境配对检验p值")
print("5. Samples: 用于该指标检验的有效样本量")
print("=" * 80)
print("{:<6} | {:<10} | {:<12} | {:<10} | {:<10} | {:<8}".format(
    "指标", "正态性", "Friedman_p", "p(A vs B)", "p(A vs C)", "样本量"))
print("-" * 80)

for metric in ['TST', 'SE', 'SOL', 'N3%', 'REM%', 'NoA']:
    data_pivot = metrics_df.pivot(index='Subject', columns='Environment', values=metric)

    # 删除含NaN的行
    valid_subjects = data_pivot.dropna().index
    sample_size = len(valid_subjects)
    data_pivot = data_pivot.dropna()

    # 如果样本太少，跳过统计检验
    if len(data_pivot) < 3:
        friedman_p = np.nan
        p_ab = np.nan
        p_ac = np.nan
        normal = False
    else:
        # 检验是否正态
        normal = all(shapiro(data_pivot[env])[1] > 0.05 for env in env_labels)

        # Friedman整体检验
        try:
            friedman_stat, friedman_p = friedmanchisquare(
                data_pivot['A'], data_pivot['B'], data_pivot['C'])
        except:
            friedman_p = np.nan

        # 配对比较 A vs B, A vs C
        if normal:
            try:
                _, p_ab = ttest_rel(data_pivot['A'], data_pivot['B'])
                _, p_ac = ttest_rel(data_pivot['A'], data_pivot['C'])
            except:
                p_ab = np.nan
                p_ac = np.nan
        else:
            try:
                _, p_ab = wilcoxon(data_pivot['A'], data_pivot['B'])
                _, p_ac = wilcoxon(data_pivot['A'], data_pivot['C'])
            except:
                p_ab = np.nan
                p_ac = np.nan

    # 格式化输出
    friedman_p_str = f"{friedman_p:.4f}" if not np.isnan(friedman_p) else "N/A"
    p_ab_str = f"{p_ab:.4f}" if not np.isnan(p_ab) else "N/A"
    p_ac_str = f"{p_ac:.4f}" if not np.isnan(p_ac) else "N/A"

    print("{:<6} | {:<10} | {:<12} | {:<10} | {:<10} | {:<8}".format(
        metric,
        str(normal),
        friedman_p_str,
        p_ab_str,
        p_ac_str,
        sample_size))

    results.append({
        'Metric': metric,
        'Normality': normal,
        'Friedman_p': friedman_p,
        'p(A vs B)': p_ab,
        'p(A vs C)': p_ac,
        'Samples': sample_size
    })

print("=" * 80)
print("提示: *p<0.05, **p<0.01, ***p<0.001")

result_df = pd.DataFrame(results)


# ======================
# 5. 可视化对比箱线图（以弹窗形式显示）
# ======================

def plot_metric(metric, metric_name):
    """绘制单指标的箱线图并以弹窗显示"""
    plt.figure(figsize=(10, 6))

    # 中文指标名称映射
    metric_names = {
        'TST': '总睡眠时间 (分钟)',
        'SE': '睡眠效率 (%)',
        'SOL': '入睡潜伏期 (分钟)',
        'N3%': '深睡眠比例 (%)',
        'REM%': 'REM睡眠比例 (%)',
        'NoA': '夜间觉醒次数'
    }

    ylabel = metric_names.get(metric, metric)

    # 箱线图
    ax = sns.boxplot(data=metrics_df, x='Environment', y=metric,
                     hue='Environment', palette='Set2', width=0.6,
                     showfliers=False, legend=False)

    # 散点图展示原始数据点
    sns.swarmplot(data=metrics_df, x='Environment', y=metric,
                  color='black', size=6, edgecolor='gray', linewidth=1)

    # 标题和标签
    plt.title(f'{ylabel} 在不同光照环境下的分布', fontsize=14, pad=15)
    plt.xlabel("光照环境", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # 添加统计显著性标记
    if not result_df.loc[result_df['Metric'] == metric].empty:
        stat_row = result_df.loc[result_df['Metric'] == metric].iloc[0]
        if not np.isnan(stat_row['p(A vs B)']):
            if stat_row['p(A vs B)'] < 0.05:
                y_max = metrics_df[metric].max()
                y_span = (metrics_df[metric].max() - metrics_df[metric].min()) * 0.1
                plt.plot([0, 1], [y_max + y_span, y_max + y_span], color='black')
                plt.text(0.5, y_max + y_span * 1.1,
                         '*p<0.05' if stat_row['p(A vs B)'] < 0.05 else 'n.s.',
                         ha='center', fontsize=12)

        if not np.isnan(stat_row['p(A vs C)']):
            if stat_row['p(A vs C)'] < 0.05:
                y_max = metrics_df[metric].max()
                y_span = (metrics_df[metric].max() - metrics_df[metric].min()) * 0.1
                plt.plot([0, 2], [y_max + y_span * 2, y_max + y_span * 2], color='black')
                plt.text(1.0, y_max + y_span * 2.1,
                         '*p<0.05' if stat_row['p(A vs C)'] < 0.05 else 'n.s.',
                         ha='center', fontsize=12)

    # 美化图形
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 以弹窗形式显示图形
    plt.show()


# 依次绘制所有指标的图形
for metric in ['TST', 'SE', 'SOL', 'N3%', 'REM%', 'NoA']:
    plot_metric(metric, metric)

    # 保存到文件
    # plt.savefig(f'./figure/{metric}_对比图.jpg', dpi=500, bbox_inches='tight')
    # plt.close()  # 关闭当前图形以节省内存

# ======================
# 6. 导出处理结果
# ======================

metrics_df.to_csv('./output/睡眠指标结果表.csv', index=False)
result_df.to_csv('./output/统计检验结果.csv', index=False)

print("\n处理完成! 结果已保存到 output 文件夹")