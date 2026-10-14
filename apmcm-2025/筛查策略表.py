# -*- coding: utf-8 -*-
"""
疾病筛查策略表生成模块
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体，确保显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class ScreeningStrategyTable:
    def __init__(self):
        self.output_dir = "筛查策略表"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_screening_strategy_flowchart(self):
        """创建分年龄段疾病筛查策略表"""
        print("正在创建筛查策略流程图...")
        
        # 筛查策略核心数据
        screening_data = {
            '年龄组': ['18-35岁', '35-50岁', '50-65岁', '65岁以上'],
            '心脏病筛查频率': ['5年一次', '3年一次', '每年', '每6个月'],
            '中风筛查频率': ['不需要', '5年一次', '2年一次', '每年'],
            '肝硬化筛查频率': ['不需要', '3年一次', '每年', '每6个月'],
            '重点筛查项目': [
                '基础体检',
                '血压、血脂、血糖',
                '心电图、颈动脉超声',
                '全面心血管评估'
            ]
        }
        
        # 转换为DataFrame
        df_screening = pd.DataFrame(screening_data)
        
        # 绘制筛查策略表格（优化颜色和样式）
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')  # 紧凑布局
        ax.axis('off')    # 隐藏坐标轴
        
        # 创建表格并设置样式
        table = ax.table(
            cellText=df_screening.values,
            colLabels=df_screening.columns,
            cellLoc='center',  # 单元格内容居中
            loc='center',      # 表格位置
            bbox=[0, 0, 1, 1]  # 表格占满整个区域
        )
        
        # 调整表格样式
        table.auto_set_font_size(False)  # 禁用自动字体大小
        table.set_fontsize(16)           # 设置字体大小
        table.scale(1, 2)                # 缩放表格（宽度1倍，高度2倍）
        
        # 设置表头样式（深蓝色背景，白色文字）
        header_color = '#2c3e50'
        for i in range(len(df_screening.columns)):
            table[(0, i)].set_facecolor(header_color)
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=20)
        
        # 设置表格内容行样式（交替颜色）
        row_colors = ['#98baddff', "#d6eafcff"]  # 浅灰、更浅灰
        for i in range(1, len(df_screening) + 1):
            for j in range(len(df_screening.columns)):
                table[(i, j)].set_facecolor(row_colors[i % 2])  # 交替颜色
                table[(i, j)].set_text_props(fontsize=16)
        
        # 添加标题
        plt.title('分年龄段疾病筛查策略', fontsize=30, fontweight='bold', pad=20)
        
        # 保存表格图片
        plt.savefig(f"{self.output_dir}/分年龄段疾病筛查策略表.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"筛查策略表已保存到 {self.output_dir} 文件夹")
        return df_screening

# 运行生成筛查策略表
if __name__ == "__main__":
    screening_table = ScreeningStrategyTable()
    screening_data = screening_table.create_screening_strategy_flowchart()
    print("\n筛查策略表数据：")
    print(screening_data)