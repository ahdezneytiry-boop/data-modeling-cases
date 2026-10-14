import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse

# ============================================
# 1. 样式配置 (严格参考 main.py 的珊瑚红美学)
# ============================================
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei"],
    "axes.unicode_minus": False,
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "axes.facecolor": "#fefaf9",  # 浅肉粉色背景
    "axes.edgecolor": "#ffffff",
    "axes.titleweight": "bold",
    "axes.titlecolor": "#00092b",  # 珊瑚红标题
    "axes.labelcolor": "#000830",
    "xtick.color": "#01065d",
    "ytick.color": "#0B0352FF",
    "text.color": "#00175d",
})

# ============================================
# 2. 数据处理
# ============================================
file_path = r"C:\Users\ahdezneytiry-boop1\AppData\Local\Programs\Microsoft VS Code\问题二_最终结果\测试集预测结果.xlsx"
df = pd.read_excel(file_path)

# 选取用于降维的特征列
features = ['HDL', 'TC', 'LDL', '血尿酸', 'TG', '血瘀质']
X = df[features]
y = df['真实标签'] # 0 为未确诊，1 为确诊

# 标准化数据
X_scaled = StandardScaler().fit_transform(X)

# PCA 降维到 2 维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Label': y.map({0: '未确诊人群', 1: '确诊高血脂人群'})
})

# ============================================
# 3. 绘图实施 (复刻投影标记风格)
# ============================================
fig, ax = plt.subplots()

# 定义颜色 (珊瑚红 vs 蓝灰色)
colors = {'未确诊人群': "#93c3ce", '确诊高血脂人群': "#3333d4"}

x_min = pca_df["PC1"].min() - 0.5
y_min = pca_df["PC2"].min() - 0.5

for label in pca_df["Label"].unique():
    subset = pca_df[pca_df["Label"] == label]
    color = colors[label]
    
    # A. 绘制核心散点
    ax.scatter(subset["PC1"], subset["PC2"], label=label, 
               color=color, s=80, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    # B. 绘制底部横轴投影 (Rug Plot 风格)
    ax.scatter(subset["PC1"], [y_min] * len(subset), 
               marker="|", color=color, alpha=0.4, s=200)
    
    # C. 绘制左侧纵轴投影
    ax.scatter([x_min] * len(subset), subset["PC2"], 
               marker="_", color=color, alpha=0.4, s=200)

    # D. 绘制分布置信椭圆 (增加图形的圆润感和显著性)
    cov = np.cov(subset["PC1"], subset["PC2"])
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(np.mean(subset["PC1"]), np.mean(subset["PC2"])),
                  width=lambda_[0]*3, height=lambda_[1]*3,
                  angle=np.rad2deg(np.arccos(v[0,0])),
                  color=color, alpha=0.1)
    ax.add_artist(ell)

# ============================================
# 4. 细节微调 (收缩轴范围，减少空白)
# ============================================
ax.set_xlabel(f"第一主成分 (解释度 {pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"第二主成分 (解释度 {pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("基于 PCA 降维的高血脂人群特征分布聚类")

# 紧凑坐标轴
ax.set_xlim(x_min, pca_df["PC1"].max() + 0.5)
ax.set_ylim(y_min, pca_df["PC2"].max() + 0.5)

ax.legend(frameon=True, facecolor='white', edgecolor="#a6a1b3", loc='upper right')
ax.grid(True, linestyle='--', alpha=0.2, color="#48475e")

plt.tight_layout()
plt.show()