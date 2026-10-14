# -*- coding: utf-8 -*-
"""
中风预测模型构建（从原代码中提取）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class StrokePredictionModel:
    def __init__(self):
        self.output_dir = "中风预测模型结果"
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.dataset = None
        self.results = None
        self.best_model = None
        
    def load_and_prepare_data(self):
        """加载和准备中风数据"""
        print("正在加载和准备中风数据...")
        
        # 加载数据
        stroke_df = pd.read_csv("data/stroke.csv")
        
        # 数据预处理
        self.dataset = self.preprocess_stroke_data(stroke_df)
        
        print("数据准备完成！")
    
    def preprocess_stroke_data(self, df):
        """预处理中风数据"""
        df = df.copy()
        
        # 处理BMI缺失值
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        
        # 编码分类变量
        le = LabelEncoder()
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))
        
        # 删除ID列
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        return df
    
    def feature_selection(self, X, y):
        """特征选择"""
        print("\n正在进行中风数据集的特征选择...")
        
        # 方法1：基于统计的特征选择
        selector_stats = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
        X_selected_stats = selector_stats.fit_transform(X, y)
        selected_features_stats = X.columns[selector_stats.get_support()].tolist()
        
        # 方法2：基于随机森林的特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(min(15, len(feature_importance)))
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性')
        plt.title('中风数据集特征重要性排序')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/中风_特征重要性.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 选择前10个重要特征
        selected_features_rf = feature_importance.head(10)['feature'].tolist()
        
        print(f"统计方法选择的特征: {selected_features_stats}")
        print(f"随机森林选择的特征: {selected_features_rf}")
        
        return selected_features_rf, feature_importance
    
    def build_models(self, X_train, X_test, y_train, y_test):
        """构建多种模型"""
        print("\n正在构建中风数据集的预测模型...")
        
        # 定义模型
        models = {
            '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
            '梯度提升': GradientBoostingClassifier(random_state=42),
            '支持向量机': SVC(probability=True, random_state=42),
            '神经网络': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # 数据标准化（对某些模型）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for name, model in models.items():
            print(f"训练{name}模型...")
            
            # 选择是否使用标准化数据
            if name in ['逻辑回归', '支持向量机', '神经网络']:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test
            
            # 训练模型
            model.fit(X_tr, y_train)
            
            # 预测
            y_pred = model.predict(X_te)
            y_pred_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # 计算AUC（二分类问题）
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} - 准确率: {accuracy:.4f}, F1分数: {f1:.4f}, 交叉验证: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        return results, scaler
    
    def plot_model_comparison(self, results):
        """绘制模型比较图"""
        print("\n正在绘制中风数据集的模型比较图...")
        
        # 准备数据
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['准确率', '精确率', '召回率', 'F1分数']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('中风数据集模型性能比较', fontsize=16, fontweight='bold')
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            row, col = i // 2, i % 2
            
            values = [results[name][metric] for name in model_names]
            bars = axes[row, col].bar(model_names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'purple'])
            
            axes[row, col].set_title(f'{metric_name}比较')
            axes[row, col].set_ylabel(metric_name)
            axes[row, col].set_ylim(0, 1)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/中风_模型性能比较.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, results, y_test):
        """绘制ROC曲线"""
        print("\n正在绘制中风数据集的ROC曲线...")
        
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                auc = result['auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('中风数据集ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/中风_ROC曲线.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, results, y_test):
        """绘制混淆矩阵"""
        print("\n正在绘制中风数据集的混淆矩阵...")
        
        n_models = len(results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('中风数据集混淆矩阵', fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (name, result) in enumerate(results.items()):
            if i < len(axes):
                cm = confusion_matrix(y_test, result['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{name}')
                axes[i].set_xlabel('预测标签')
                axes[i].set_ylabel('真实标签')
        
        # 隐藏多余的子图
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/中风_混淆矩阵.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curves(self, best_model, X, y, model_name):
        """绘制学习曲线"""
        print(f"\n正在绘制中风数据集{model_name}的学习曲线...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            best_model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练集得分')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='验证集得分')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('训练样本数')
        plt.ylabel('准确率')
        plt.title(f'中风数据集{model_name}学习曲线')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/中风_{model_name}_学习曲线.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def hyperparameter_tuning(self, X_train, y_train):
        """超参数调优"""
        print("\n正在进行中风数据集的超参数调优...")
        
        # 随机森林超参数调优
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        print(f"随机森林最佳参数: {rf_grid.best_params_}")
        print(f"随机森林最佳得分: {rf_grid.best_score_:.4f}")
        
        return rf_grid.best_estimator_
    
    def handle_imbalanced_data(self, X_train, y_train):
        """处理不平衡数据"""
        print("\n正在处理中风数据集的数据不平衡问题...")
        
        # 检查类别分布
        class_counts = pd.Series(y_train).value_counts()
        print(f"原始类别分布: {class_counts.to_dict()}")
        
        # 中风数据通常不平衡，使用SMOTE
        if len(class_counts) == 2 and min(class_counts) / max(class_counts) < 0.5:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            new_class_counts = pd.Series(y_resampled).value_counts()
            print(f"SMOTE后类别分布: {new_class_counts.to_dict()}")
            
            return X_resampled, y_resampled
        else:
            print("数据相对平衡，无需处理")
            return X_train, y_train
    
    def analyze_stroke_dataset(self):
        """分析中风数据集"""
        print("\n" + "="*50)
        print("开始分析中风数据集")
        print("="*50)
        
        df = self.dataset
        
        # 确定特征和目标变量
        target_col = 'stroke'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"目标变量分布: {y.value_counts().to_dict()}")
        
        # 特征选择
        selected_features, feature_importance = self.feature_selection(X, y)
        X_selected = X[selected_features]
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 处理不平衡数据
        X_train_balanced, y_train_balanced = self.handle_imbalanced_data(X_train, y_train)
        
        # 构建模型
        results, scaler = self.build_models(X_train_balanced, X_test, y_train_balanced, y_test)
        
        # 超参数调优
        best_rf = self.hyperparameter_tuning(X_train_balanced, y_train_balanced)
        
        # 绘制各种图表
        self.plot_model_comparison(results)
        self.plot_roc_curves(results, y_test)
        self.plot_confusion_matrices(results, y_test)
        
        # 找到最佳模型
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
        best_model = results[best_model_name]['model']
        
        print(f"\n中风数据集最佳模型: {best_model_name}")
        print(f"最佳模型F1分数: {results[best_model_name]['f1']:.4f}")
        
        # 绘制学习曲线
        self.plot_learning_curves(best_model, X_train_balanced, y_train_balanced, best_model_name)
        
        # 保存结果
        self.results = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'results': results,
            'feature_importance': feature_importance,
            'selected_features': selected_features
        }
        
        return self.results
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*60)
        print("中风预测模型总结报告")
        print("="*60)
        
        if not self.results:
            print("没有结果可生成报告")
            return
        
        best_model_name = self.results['best_model_name']
        best_result = self.results['results'][best_model_name]
        
        summary_data = [{
            '数据集': 'stroke',
            '最佳模型': best_model_name,
            '准确率': f"{best_result['accuracy']:.4f}",
            '精确率': f"{best_result['precision']:.4f}",
            '召回率': f"{best_result['recall']:.4f}",
            'F1分数': f"{best_result['f1']:.4f}",
            'AUC': f"{best_result['auc']:.4f}" if best_result['auc'] else "N/A",
            '交叉验证': f"{best_result['cv_mean']:.4f}±{best_result['cv_std']:.4f}"
        }]
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # 保存总结报告
        summary_df.to_csv(f"{self.output_dir}/中风模型性能总结.csv", index=False, encoding='utf-8-sig')
        
        # 绘制最佳模型性能图
        plt.figure(figsize=(10, 6))
        metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
        values = [
            best_result['accuracy'],
            best_result['precision'],
            best_result['recall'],
            best_result['f1'],
            best_result['auc'] if best_result['auc'] else 0
        ]
        
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'purple'])
        plt.title(f'中风最佳模型({best_model_name})性能指标')
        plt.ylim(0, 1)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/中风最佳模型性能指标.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始中风预测模型构建")
        
        # 加载和准备数据
        self.load_and_prepare_data()
        
        # 分析中风数据集
        self.analyze_stroke_dataset()
        
        # 生成总结报告
        self.generate_summary_report()
        
        print(f"\n中风预测模型分析完成！所有图表已保存到 {self.output_dir} 文件夹中。")
        
        return self.results

if __name__ == "__main__":
    stroke_analyzer = StrokePredictionModel()
    stroke_analyzer.run_analysis()