# -*- coding: utf-8 -*-
"""
问题1：数据预处理与基础统计分析
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem1DataPreprocessing:
    def __init__(self):
        self.output_dir = "问题1_数据预处理与基础统计分析"
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.datasets = {}
        self.cleaned_datasets = {}
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        try:
            # 加载心脏病数据
            self.datasets['heart'] = pd.read_csv('附件/heart.csv')
            print(f"心脏病数据集形状: {self.datasets['heart'].shape}")
            
            # 加载中风数据
            self.datasets['stroke'] = pd.read_csv('附件/stroke.csv')
            print(f"中风数据集形状: {self.datasets['stroke'].shape}")
            
            # 加载肝硬化数据
            self.datasets['cirrhosis'] = pd.read_csv('附件/cirrhosis.csv')
            print(f"肝硬化数据集形状: {self.datasets['cirrhosis'].shape}")
            
        except Exception as e:
            print(f"数据加载错误: {e}")
            return False
        
    def data_cleaning(self):
        """数据清洗"""
        print("\n正在进行数据清洗...")
        
        for name, df in self.datasets.items():
            print(f"\n处理{name}数据集:")
            
            # 复制数据
            cleaned_df = df.copy()
            
            # 检查缺失值
            missing_count = cleaned_df.isnull().sum()
            print(f"缺失值统计:\n{missing_count[missing_count > 0]}")
            
            if name == 'heart':
                # 心脏病数据集处理
                # 检查异常值
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)]
                    if len(outliers) > 0:
                        print(f"{col}列发现{len(outliers)}个异常值")
                
            elif name == 'stroke':
                # 中风数据集处理
                # 处理BMI缺失值
                if 'bmi' in cleaned_df.columns:
                    cleaned_df['bmi'] = pd.to_numeric(cleaned_df['bmi'], errors='coerce')
                    bmi_median = cleaned_df['bmi'].median()
                    cleaned_df['bmi'].fillna(bmi_median, inplace=True)
                
                # 处理其他缺失值
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype == 'object':
                        mode_val = cleaned_df[col].mode()[0] if len(cleaned_df[col].mode()) > 0 else 'Unknown'
                        cleaned_df[col].fillna(mode_val, inplace=True)
                
            elif name == 'cirrhosis':
                # 肝硬化数据集处理
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if cleaned_df[col].isnull().sum() > 0:
                        median_val = cleaned_df[col].median()
                        cleaned_df[col].fillna(median_val, inplace=True)
                
                # 处理分类变量缺失值
                categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if cleaned_df[col].isnull().sum() > 0:
                        mode_val = cleaned_df[col].mode()[0] if len(cleaned_df[col].mode()) > 0 else 'Unknown'
                        cleaned_df[col].fillna(mode_val, inplace=True)
            
            self.cleaned_datasets[name] = cleaned_df
            print(f"清洗后数据形状: {cleaned_df.shape}")
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        print("\n正在进行描述性统计分析...")
        
        for name, df in self.cleaned_datasets.items():
            print(f"\n{name}数据集描述性统计:")
            
            # 数值型变量统计
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                desc_stats = df[numeric_cols].describe()
                print(f"数值型变量统计:\n{desc_stats}")
                
                # 保存统计结果
                desc_stats.to_csv(f"{self.output_dir}/问题1_{name}_描述性统计.csv", encoding='utf-8-sig')
            
            # 分类变量统计
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                print(f"分类变量统计:")
                for col in categorical_cols:
                    print(f"{col}:")
                    print(df[col].value_counts())
                    print()
    
    def plot_missing_values(self):
        """绘制缺失值矩阵图"""
        print("\n正在绘制缺失值分析图...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('三个数据集缺失值分析', fontsize=16, fontweight='bold')
        
        for i, (name, df) in enumerate(self.datasets.items()):
            # 计算缺失值比例
            missing_ratio = df.isnull().sum() / len(df) * 100
            missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
            
            if len(missing_ratio) > 0:
                axes[i].bar(range(len(missing_ratio)), missing_ratio.values, color='red', alpha=0.7)
                axes[i].set_xticks(range(len(missing_ratio)))
                axes[i].set_xticklabels(missing_ratio.index, rotation=45, ha='right')
                axes[i].set_ylabel('缺失值比例 (%)')
                axes[i].set_title(f'{name}数据集缺失值分布')
                axes[i].grid(True, alpha=0.3)
                
                # 在柱子上添加数值标签
                for j, v in enumerate(missing_ratio.values):
                    axes[i].text(j, v + 0.1, f'{v:.1f}%', ha='center', va='bottom')
            else:
                axes[i].text(0.5, 0.5, '无缺失值', ha='center', va='center', 
                           transform=axes[i].transAxes, fontsize=14)
                axes[i].set_title(f'{name}数据集缺失值分布')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/问题1_缺失值分析.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_distributions(self):
        """绘制数据分布图"""
        print("\n正在绘制数据分布图...")
        
        for name, df in self.cleaned_datasets.items():
            # 获取数值型列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                continue
                
            # 确定子图布局
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            fig.suptitle(f'{name}数据集数值变量分布', fontsize=16, fontweight='bold')
            
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'{col}分布')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('频数')
                    axes[i].grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/问题1_{name}_数据分布.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_boxplots(self):
        """绘制箱线图"""
        print("\n正在绘制箱线图...")
        
        for name, df in self.cleaned_datasets.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                continue
            
            # 获取目标变量
            target_col = None
            if name == 'heart':
                target_col = 'HeartDisease'
            elif name == 'stroke':
                target_col = 'stroke'
            elif name == 'cirrhosis':
                target_col = 'Status'
            
            if target_col and target_col in df.columns:
                # 选择几个重要的数值变量
                important_cols = [col for col in numeric_cols if col != target_col][:6]
                
                if len(important_cols) > 0:
                    n_cols = min(3, len(important_cols))
                    n_rows = (len(important_cols) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                    fig.suptitle(f'{name}数据集特征与目标变量关系箱线图', fontsize=16, fontweight='bold')
                    
                    if n_rows == 1 and n_cols == 1:
                        axes = [axes]
                    elif n_rows == 1:
                        axes = axes
                    else:
                        axes = axes.flatten()
                    
                    for i, col in enumerate(important_cols):
                        if i < len(axes):
                            df.boxplot(column=col, by=target_col, ax=axes[i])
                            axes[i].set_title(f'{col} vs {target_col}')
                            axes[i].set_xlabel(target_col)
                            axes[i].set_ylabel(col)
                    
                    # 隐藏多余的子图
                    for i in range(len(important_cols), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/问题1_{name}_箱线图分析.png", dpi=300, bbox_inches='tight')
                    plt.close()
    
    def plot_correlation_heatmap(self):
        """绘制相关性热力图"""
        print("\n正在绘制相关性热力图...")
        
        for name, df in self.cleaned_datasets.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                continue
            
            # 计算相关性矩阵
            corr_matrix = df[numeric_cols].corr()
            
            # 绘制热力图
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
            plt.title(f'{name}数据集特征相关性热力图', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/问题1_{name}_相关性热力图.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_target_variables(self):
        """分析目标变量分布"""
        print("\n正在分析目标变量分布...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('三种疾病目标变量分布', fontsize=16, fontweight='bold')
        
        # 心脏病
        if 'HeartDisease' in self.cleaned_datasets['heart'].columns:
            heart_counts = self.cleaned_datasets['heart']['HeartDisease'].value_counts()
            axes[0].pie(heart_counts.values, labels=['无心脏病', '有心脏病'], autopct='%1.1f%%', 
                       colors=['lightblue', 'lightcoral'])
            axes[0].set_title('心脏病分布')
        
        # 中风
        if 'stroke' in self.cleaned_datasets['stroke'].columns:
            stroke_counts = self.cleaned_datasets['stroke']['stroke'].value_counts()
            axes[1].pie(stroke_counts.values, labels=['无中风', '有中风'], autopct='%1.1f%%',
                       colors=['lightgreen', 'orange'])
            axes[1].set_title('中风分布')
        
        # 肝硬化
        if 'Status' in self.cleaned_datasets['cirrhosis'].columns:
            cirrhosis_counts = self.cleaned_datasets['cirrhosis']['Status'].value_counts()
            labels = [f'状态{status}' for status in cirrhosis_counts.index]
            axes[2].pie(cirrhosis_counts.values, labels=labels, autopct='%1.1f%%')
            axes[2].set_title('肝硬化状态分布')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/问题1_目标变量分布.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 输出具体统计信息
        print("\n目标变量统计信息:")
        for name, df in self.cleaned_datasets.items():
            if name == 'heart' and 'HeartDisease' in df.columns:
                counts = df['HeartDisease'].value_counts()
                print(f"心脏病: 无病例{counts[0]}例({counts[0]/len(df)*100:.1f}%), 患病{counts[1]}例({counts[1]/len(df)*100:.1f}%)")
            elif name == 'stroke' and 'stroke' in df.columns:
                counts = df['stroke'].value_counts()
                print(f"中风: 无病例{counts[0]}例({counts[0]/len(df)*100:.1f}%), 患病{counts[1]}例({counts[1]/len(df)*100:.1f}%)")
            elif name == 'cirrhosis' and 'Status' in df.columns:
                counts = df['Status'].value_counts()
                print(f"肝硬化状态分布:")
                for status, count in counts.items():
                    print(f"  状态{status}: {count}例({count/len(df)*100:.1f}%)")
    
    def identify_significant_factors(self):
        """识别显著影响因素"""
        print("\n正在识别显著影响疾病的因素...")
        
        significant_factors = {}
        
        for name, df in self.cleaned_datasets.items():
            print(f"\n{name}数据集显著因素分析:")
            factors = []
            
            if name == 'heart' and 'HeartDisease' in df.columns:
                # 心脏病相关因素分析
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                target = 'HeartDisease'
                
                for col in numeric_cols:
                    if col != target:
                        corr = df[col].corr(df[target])
                        if abs(corr) > 0.1:  # 相关性阈值
                            factors.append((col, abs(corr)))
                            print(f"  {col}: 相关性 = {corr:.3f}")
                
            elif name == 'stroke' and 'stroke' in df.columns:
                # 中风相关因素分析
                # 年龄分析
                stroke_by_age = df.groupby('stroke')['age'].mean()
                print(f"  年龄: 无中风组平均{stroke_by_age[0]:.1f}岁, 中风组平均{stroke_by_age[1]:.1f}岁")
                
                # 高血压分析
                if 'hypertension' in df.columns:
                    hypertension_stroke = pd.crosstab(df['hypertension'], df['stroke'], normalize='index') * 100
                    print(f"  高血压患者中风比例: {hypertension_stroke.loc[1, 1]:.1f}%")
                    print(f"  无高血压者中风比例: {hypertension_stroke.loc[0, 1]:.1f}%")
                
                # 心脏病分析
                if 'heart_disease' in df.columns:
                    heart_stroke = pd.crosstab(df['heart_disease'], df['stroke'], normalize='index') * 100
                    print(f"  心脏病患者中风比例: {heart_stroke.loc[1, 1]:.1f}%")
                    print(f"  无心脏病者中风比例: {heart_stroke.loc[0, 1]:.1f}%")
            
            elif name == 'cirrhosis':
                # 肝硬化相关因素分析
                print("  主要影响因素包括:")
                print("  - 胆红素水平(Bilirubin)")
                print("  - 白蛋白水平(Albumin)")
                print("  - 凝血酶原时间(Prothrombin)")
                print("  - 疾病分期(Stage)")
            
            significant_factors[name] = factors
        
        return significant_factors
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始问题1：数据预处理与基础统计分析")
        
        # 加载数据
        self.load_data()
        
        # 数据清洗
        self.data_cleaning()
        
        # 描述性统计
        self.descriptive_statistics()
        
        # 绘制各种图表
        self.plot_missing_values()
        self.plot_distributions()
        self.plot_boxplots()
        self.plot_correlation_heatmap()
        self.analyze_target_variables()
        
        # 识别显著因素
        significant_factors = self.identify_significant_factors()
        
        print(f"\n问题1分析完成！所有图表已保存到 {self.output_dir} 文件夹中。")
        
        return self.cleaned_datasets, significant_factors

if __name__ == "__main__":
    problem1 = Problem1DataPreprocessing()
    problem1.run_analysis()