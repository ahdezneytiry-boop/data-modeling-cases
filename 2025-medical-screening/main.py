# -*- coding: utf-8 -*-
"""
多疾病筛查案例
B题：疾病的预测与大数据分析
主程序文件
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入各个问题的模块
from problem1_data_preprocessing import Problem1DataPreprocessing
from problem2_model_building import Problem2ModelBuilding
from problem3_multi_disease_analysis import Problem3MultiDiseaseAnalysis
from problem4_prevention_recommendations import Problem4PreventionRecommendations

def main():
    """主函数"""
    print("="*80)
    print("多疾病筛查案例")
    print("B题：疾病的预测与大数据分析")
    print("="*80)
    
    # 创建输出目录
    output_dirs = [
        "问题1_数据预处理与基础统计分析",
        "问题2_不同疾病预测模型的构建", 
        "问题3_多疾病关联与综合风险评估",
        "问题4_预防建议和措施"
    ]
    
    for dir_name in output_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    try:
        # 问题1：数据预处理与基础统计分析
        print("\n" + "="*50)
        print("开始执行问题1：数据预处理与基础统计分析")
        print("="*50)
        problem1 = Problem1DataPreprocessing()
        problem1.run_analysis()
        
        # 问题2：不同疾病预测模型的构建
        print("\n" + "="*50)
        print("开始执行问题2：不同疾病预测模型的构建")
        print("="*50)
        problem2 = Problem2ModelBuilding()
        problem2.run_analysis()
        
        # 问题3：多疾病关联与综合风险评估
        print("\n" + "="*50)
        print("开始执行问题3：多疾病关联与综合风险评估")
        print("="*50)
        problem3 = Problem3MultiDiseaseAnalysis()
        problem3.run_analysis()
        
        # 问题4：预防建议和措施
        print("\n" + "="*50)
        print("开始执行问题4：预防建议和措施")
        print("="*50)
        problem4 = Problem4PreventionRecommendations()
        problem4.run_analysis()
        
        print("\n" + "="*80)
        print("所有分析完成！请查看各个问题文件夹中的图表和结果。")
        print("="*80)
        
    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()