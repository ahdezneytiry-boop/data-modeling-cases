import plotly.graph_objects as go
import numpy as np
import pandas as pd

# 模拟数据 (请替换为处理后数据.csv中的真实列)
n_samples = 200
# 假设我们要展示 TG (X), BMI (Y), 血瘀质分值 (Z)
tg = np.random.uniform(0.5, 5.0, n_samples)
bmi = np.random.uniform(18, 35, n_samples)
stasis = np.random.uniform(10, 80, n_samples)
# 风险得分（基于你的第一问模型预测出的概率）
risk_score = 1 / (1 + np.exp(-(0.8*tg + 0.3*(bmi-24) + 0.1*stasis - 4)))

fig = go.Figure(data=[go.Scatter3d(
    x=tg, y=bmi, z=stasis,
    mode='markers',
    marker=dict(
        size=risk_score * 20,          # 风险越高，气泡越大
        color=risk_score,               # 风险越高，颜色越深
        colorscale='Pinkyl',            # 使用高级粉色渐变
        opacity=0.8,
        line=dict(width=0.5, color='white')
    ),
    text=[f"风险概率: {r:.2f}" for r in risk_score],
    hoverinfo='text+x+y+z'
)])

# 设置坐标轴标签（根据第一问结论）
fig.update_layout(
    title="中老年高血脂风险多维景观分布图 (TG-BMI-血瘀质)",
    scene=dict(
        xaxis_title='甘油三酯 (TG)',
        yaxis_title='体质指数 (BMI)',
        zaxis_title='血瘀质得分',
        xaxis=dict(backgroundcolor="rgb(250, 250, 250)", gridcolor="white", showbackground=True),
        yaxis=dict(backgroundcolor="rgb(250, 250, 250)", gridcolor="white", showbackground=True),
        zaxis=dict(backgroundcolor="rgb(250, 250, 250)", gridcolor="white", showbackground=True)
    ),
    width=1000, height=800,
    margin=dict(r=20, b=10, l=10, t=50)
)

# fig.write_html("3D_Risk_Landscape.html")
fig.show()