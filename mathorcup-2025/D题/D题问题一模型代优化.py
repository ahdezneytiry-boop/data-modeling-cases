#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
def load_data():
    # 假设数据已加载为DataFrame
    df_hist = pd.read_excel('附件2.xlsx')  # 历史10分钟颗粒度数据
    df_known = pd.read_excel('附件3.xlsx')  # 预知货量
    df_routes = pd.read_excel('附件1.xlsx')  # 线路信息
    
    # 转换时间格式（修正错误的关键步骤）
    if not pd.api.types.is_datetime64_any_dtype(df_hist['分钟起始']):
        df_hist['分钟起始'] = pd.to_datetime(df_hist['分钟起始']).dt.time
    
    # 合并数据
    df_hist = pd.merge(df_hist, df_routes, on='线路编码', how='left')
    return df_hist, df_known

# 2. 特征工程（修正版）
def create_features(df):
    # 直接从time对象提取特征
    df['小时'] = df['分钟起始'].apply(lambda x: x.hour)
    df['分钟'] = df['分钟起始'].apply(lambda x: x.minute)
    df['是否高峰时段'] = df['小时'].apply(lambda x: 1 if x in [6, 14] else 0)
    
    # 线路特征标准化
    if '在途时长' in df.columns:
        df['在途时长_norm'] = (df['在途时长'] - df['在途时长'].mean()) / df['在途时长'].std()
    return df

# 3. 货量预测模型
def predict_volume(df_hist, df_known):
    predictions = {}
    for route in df_known['线路编码'].unique():
        # 获取线路数据
        route_data = df_hist[df_hist['线路编码'] == route].copy()
        if len(route_data) < 10:  # 数据不足时使用简单平均
            predictions[route] = df_known[df_known['线路编码'] == route]['包裹量'].mean()
            continue
            
        # 时间序列预测
        try:
            ts_model = ARIMA(route_data['包裹量'], order=(1,1,1))
            ts_result = ts_model.fit()
            forecast = ts_result.forecast(steps=1)[0]
        except:
            forecast = route_data['包裹量'].mean()
        
        # 机器学习预测
        X = route_data[['小时', '分钟', '是否高峰时段', '在途时长_norm']]
        y = route_data['包裹量']
        if len(X) > 20:  # 足够数据才训练模型
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            rf = RandomForestRegressor(n_estimators=50)
            rf.fit(X_train, y_train)
            ml_pred = rf.predict(X_test.mean().values.reshape(1,-1))[0]
        else:
            ml_pred = y.mean()
        
        # 融合预测
        final_pred = 0.7 * forecast + 0.3 * ml_pred
        predictions[route] = max(0, round(final_pred))
    
    return predictions

# 4. 10分钟颗粒度拆解
def disaggregate_to_10min(total_volume, route_data):
    # 获取该线路的时间分布
    time_dist = route_data.groupby('分钟起始')['包裹量'].mean()
    time_dist = time_dist / time_dist.sum()
    
    # 生成完整时间区间
    time_slots = pd.date_range("00:00", "23:50", freq="10min").time
    time_dist = time_dist.reindex(time_slots, fill_value=0)
    time_dist = time_dist / time_dist.sum()  # 重新归一化
    
    # 按比例分配
    disaggregated = (time_dist * total_volume).round().astype(int)
    diff = total_volume - disaggregated.sum()
    if diff != 0:
        max_idx = disaggregated.idxmax()
        disaggregated.loc[max_idx] += diff
    return disaggregated

# 主执行流程
if __name__ == "__main__":
    # 加载数据
    df_hist, df_known = load_data()
    df_hist = create_features(df_hist)
    
    # 预测总货量
    predictions = predict_volume(df_hist, df_known)
    
    # 示例：拆解特定线路
    target_route = "场地3 - 站点83 - 0600"
    route_data = df_hist[df_hist['线路编码'] == target_route]
    disagg_result = disaggregate_to_10min(predictions[target_route], route_data)
    
    # 可视化
    plt.figure(figsize=(12, 5))
    disagg_result.plot(kind='bar', color='#1f77b4')
    plt.title(f'线路【{target_route}】10分钟颗粒度货量分布预测', fontsize=14)
    plt.xlabel('时间区间', fontsize=12)
    plt.ylabel('包裹量', fontsize=12)
    plt.xticks(rotation=45, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # 输出结果表示例
    print(f"\n线路 {target_route} 预测结果:")
    print(f"- 预测总货量: {predictions[target_route]}")
    print("- 前6个时间区间分配:")
    print(disagg_result.head(6))


# In[ ]:




