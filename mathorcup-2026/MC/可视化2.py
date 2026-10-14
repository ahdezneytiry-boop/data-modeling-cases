import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
file_path = r"C:\Users\ahdezneytiry-boop1\AppData\Local\Programs\Microsoft VS Code\问题二_最终结果\测试集预测结果.xlsx"
df = pd.read_excel(file_path)

# 获取真实标签和预测结果
y_true = df['真实标签']
y_prob = df['预测概率']
y_pred = df['预测标签']

# 创建画布
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# ============================================
# 图左：ROC 曲线 (Receiver Operating Characteristic)
# ============================================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.05])
ax[0].set_xlabel('False Positive Rate (假阳性率)')
ax[0].set_ylabel('True Positive Rate (真阳性率)')
ax[0].set_title('ROC 曲线分析')
ax[0].legend(loc="lower right")
ax[0].grid(alpha=0.3)

# ============================================
# 图右：混淆矩阵 (Confusion Matrix)
# ============================================
cm = confusion_matrix(y_true, y_pred)
# 转换成百分比形式（可选）
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1], 
            xticklabels=['未确诊(0)', '确诊(1)'], 
            yticklabels=['未确诊(0)', '确诊(1)'],
            annot_kws={"size": 16})

# 在热图里加上百分比标注（模仿示范图的专业感）
for i in range(2):
    for j in range(2):
        ax[1].text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1%})', 
                   ha='center', va='center', color='black', fontsize=12)

ax[1].set_xlabel('预测标签')
ax[1].set_ylabel('真实标签')
ax[1].set_title('预测结果混淆矩阵')

plt.tight_layout()
plt.show()

# 保存图片用于论文
# plt.savefig("模型评价指标图.png", dpi=300)