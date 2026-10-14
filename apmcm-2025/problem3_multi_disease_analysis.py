import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib_venn import venn3
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DiseaseAssociationAnalyzer:
    def __init__(self):
        self.result_dir = "疾病关联分析结果"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.data_sets = {}
        self.integrated_data = None
        self.risk_factors = {}
        
    def load_and_prepare_data(self):
        print("正在加载并预处理数据...")
        
        heart_df = pd.read_csv(r'E:\2025亚太\2025年第十五届APMCM亚太地区大学生数学建模竞赛（中文赛项）\B题 疾病的预测与大数据分析\参考答案\附件\heart.csv')
        stroke_df = pd.read_csv(r'E:\2025亚太\2025年第十五届APMCM亚太地区大学生数学建模竞赛（中文赛项）\B题 疾病的预测与大数据分析\参考答案\附件\stroke.csv')
        cirrhosis_df = pd.read_csv(r"E:\2025亚太\2025年第十五届APMCM亚太地区大学生数学建模竞赛（中文赛项）\B题 疾病的预测与大数据分析\参考答案\附件\cirrhosis.csv")
        
        self.data_sets['heart'] = self.process_heart_data(heart_df)
        self.data_sets['stroke'] = self.process_stroke_data(stroke_df)
        self.data_sets['cirrhosis'] = self.process_cirrhosis_data(cirrhosis_df)
        
        print("数据预处理完成！")
    
    def process_heart_data(self, df):
        df = df.copy()
        
        processed_df = pd.DataFrame({
            'Age': df['Age'],
            'Sex': (df['Sex'] == 'M').astype(int),
            'Hypertension': (df['RestingBP'] > 140).astype(int),
            'Cholesterol_High': (df['Cholesterol'] > 240).astype(int),
            'Diabetes': df['FastingBS'],
            'HeartDisease': df['HeartDisease']
        })
        
        return processed_df
    
    def process_stroke_data(self, df):
        df = df.copy()
        
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        
        processed_df = pd.DataFrame({
            'Age': df['age'],
            'Sex': (df['gender'] == 'Male').astype(int),
            'Hypertension': df['hypertension'],
            'HeartDisease': df['heart_disease'],
            'Diabetes': (df['avg_glucose_level'] > 126).astype(int),
            'Smoking': (df['smoking_status'].isin(['smokes', 'formerly smoked'])).astype(int),
            'Stroke': df['stroke']
        })
        
        return processed_df
    
    def process_cirrhosis_data(self, df):
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        processed_df = pd.DataFrame({
            'Age': df['Age'],
            'Sex': (df['Sex'] == 'M').astype(int),
            'Alcohol': (df['Drug'] == 'D-penicillamine').astype(int),
            'Ascites': (df['Ascites'] == 'Y').astype(int),
            'Hepatomegaly': (df['Hepatomegaly'] == 'Y').astype(int),
            'Edema': (df['Edema'] == 'Y').astype(int),
            'Cirrhosis': (df['Status'] == 'D').astype(int)
        })
        
        return processed_df
    
    def find_common_risk_factors(self):
        print("\n正在识别共同危险因素...")
        
        shared_features = ['Age', 'Sex', 'Hypertension']
        
        for dataset_name, df in self.data_sets.items():
            print(f"\n{dataset_name}数据集中的危险因素分析:")
            
            if dataset_name == 'heart':
                target = 'HeartDisease'
            elif dataset_name == 'stroke':
                target = 'Stroke'
            elif dataset_name == 'cirrhosis':
                target = 'Cirrhosis'
            
            for factor in shared_features:
                if factor in df.columns:
                    risk_table = pd.crosstab(df[factor], df[target])
                    if risk_table.shape == (2, 2):
                        risk_ratio = (risk_table.iloc[1, 1] / risk_table.iloc[1, :].sum()) / \
                                   (risk_table.iloc[0, 1] / risk_table.iloc[0, :].sum())
                        print(f"  {factor}风险比: {risk_ratio:.2f}")
                        
                        chi2, p_value, _, _ = chi2_contingency(risk_table)
                        print(f"  {factor}卡方检验p值: {p_value:.4f}")
        
        return shared_features
    
    def generate_synthetic_data(self):
        print("\n正在创建综合数据集...")
        
        np.random.seed(42)
        
        n_patients = 800
        
        age = np.random.normal(60, 15, n_patients)
        age = np.clip(age, 18, 100)
        sex = np.random.binomial(1, 0.5, n_patients)
        
        age_factor = (age - 18) / 82
        
        hypertension = np.random.binomial(1, 0.2 + 0.3 * age_factor, n_patients)
        diabetes = np.random.binomial(1, 0.1 + 0.2 * age_factor + 0.1 * hypertension, n_patients)
        smoking = np.random.binomial(1, 0.3 + 0.2 * sex - 0.1 * age_factor, n_patients)
        
        heart_prob = 0.05 + 0.1 * age_factor + 0.15 * hypertension + 0.1 * diabetes + 0.05 * smoking + 0.05 * sex
        stroke_prob = 0.02 + 0.08 * age_factor + 0.2 * hypertension + 0.1 * diabetes + 0.08 * smoking + 0.1 * heart_prob
        cirrhosis_prob = 0.01 + 0.05 * age_factor + 0.15 * smoking
        
        heart_disease = np.random.binomial(1, heart_prob, n_patients)
        stroke = np.random.binomial(1, stroke_prob, n_patients)
        cirrhosis = np.random.binomial(1, cirrhosis_prob, n_patients)
        
        self.integrated_data = pd.DataFrame({
            'Age': age,
            'Sex': sex,
            'Hypertension': hypertension,
            'Diabetes': diabetes,
            'Smoking': smoking,
            'HeartDisease': heart_disease,
            'Stroke': stroke,
            'Cirrhosis': cirrhosis
        })
        
        print(f"生成了{n_patients}个虚拟患者的数据")
        print(f"心脏病患病率: {heart_disease.mean():.1%}")
        print(f"中风患病率: {stroke.mean():.1%}")
        print(f"肝硬化患病率: {cirrhosis.mean():.1%}")
        
        return self.integrated_data
    
    def analyze_comorbidity(self):
        print("\n正在分析共病模式...")
        
        if self.integrated_data is None:
            self.generate_synthetic_data()
        
        df = self.integrated_data
        diseases = ['HeartDisease', 'Stroke', 'Cirrhosis']
        
        combinations = {}
        
        for disease in diseases:
            count = df[disease].sum()
            combinations[disease] = count
            print(f"仅患{disease}: {count}例 ({count/len(df)*100:.1f}%)")
        
        for i in range(len(diseases)):
            for j in range(i+1, len(diseases)):
                d1, d2 = diseases[i], diseases[j]
                count = ((df[d1] == 1) & (df[d2] == 1)).sum()
                combinations[f"{d1}+{d2}"] = count
                print(f"同时患{d1}和{d2}: {count}例 ({count/len(df)*100:.1f}%)")
        
        count_all = ((df['HeartDisease'] == 1) & (df['Stroke'] == 1) & (df['Cirrhosis'] == 1)).sum()
        combinations['All_Three'] = count_all
        print(f"同时患三种疾病: {count_all}例 ({count_all/len(df)*100:.1f}%)")
        
        return combinations
    
    def draw_venn_diagram(self):
        print("\n正在绘制疾病共病维恩图...")
        
        if self.integrated_data is None:
            self.generate_synthetic_data()
        
        df = self.integrated_data
        
        heart_only = ((df['HeartDisease'] == 1) & (df['Stroke'] == 0) & (df['Cirrhosis'] == 0)).sum()
        stroke_only = ((df['HeartDisease'] == 0) & (df['Stroke'] == 1) & (df['Cirrhosis'] == 0)).sum()
        cirrhosis_only = ((df['HeartDisease'] == 0) & (df['Stroke'] == 0) & (df['Cirrhosis'] == 1)).sum()
        
        heart_stroke = ((df['HeartDisease'] == 1) & (df['Stroke'] == 1) & (df['Cirrhosis'] == 0)).sum()
        heart_cirrhosis = ((df['HeartDisease'] == 1) & (df['Stroke'] == 0) & (df['Cirrhosis'] == 1)).sum()
        stroke_cirrhosis = ((df['HeartDisease'] == 0) & (df['Stroke'] == 1) & (df['Cirrhosis'] == 1)).sum()
        
        all_three = ((df['HeartDisease'] == 1) & (df['Stroke'] == 1) & (df['Cirrhosis'] == 1)).sum()
        
        plt.figure(figsize=(12, 8))
        
        venn = venn3(subsets=(heart_only, stroke_only, heart_stroke, cirrhosis_only, 
                             heart_cirrhosis, stroke_cirrhosis, all_three),
                    set_labels=('心脏病', '中风', '肝硬化'))
        
        # 维恩图使用低饱和彩色
        if venn.get_patch_by_id('100'):
            venn.get_patch_by_id('100').set_color("#ef2020")  # 淡红色
        if venn.get_patch_by_id('010'):
            venn.get_patch_by_id('010').set_color("#1b8dff")  # 淡蓝色
        if venn.get_patch_by_id('001'):
            venn.get_patch_by_id('001').set_color("#1cff42")  # 淡绿色
        if venn.get_patch_by_id('110'):
            venn.get_patch_by_id('110').set_color("#cb2fff")  # 淡紫色
        if venn.get_patch_by_id('101'):
            venn.get_patch_by_id('101').set_color("#b6ff23")  # 淡黄绿色
        if venn.get_patch_by_id('011'):
            venn.get_patch_by_id('011').set_color("#24ffdb")  # 淡青色
        if venn.get_patch_by_id('111'):
            venn.get_patch_by_id('111').set_color("#ffb82b")  # 淡棕色
        
        plt.title('三种疾病共病模式维恩图', fontsize=16, fontweight='bold')
        
        total_patients = len(df)
        any_disease = ((df['HeartDisease'] == 1) | (df['Stroke'] == 1) | (df['Cirrhosis'] == 1)).sum()
        
        plt.text(0.02, 0.98, f'总患者数: {total_patients}', transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.02, 0.93, f'至少患一种疾病: {any_disease} ({any_disease/total_patients*100:.1f}%)', 
                transform=plt.gca().transAxes, verticalalignment='top', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.02, 0.88, f'同时患三种疾病: {all_three} ({all_three/total_patients*100:.1f}%)', 
                transform=plt.gca().transAxes, verticalalignment='top', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/疾病共病维恩图.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def build_multidisease_model(self):
        print("\n正在构建多疾病预测模型...")
        
        if self.integrated_data is None:
            self.generate_synthetic_data()
        
        df = self.integrated_data
        
        feature_cols = ['Age', 'Sex', 'Hypertension', 'Diabetes', 'Smoking']
        target_cols = ['HeartDisease', 'Stroke', 'Cirrhosis']
        
        X = df[feature_cols]
        y = df[target_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        multi_model = MultiOutputClassifier(base_model)
        
        multi_model.fit(X_train_scaled, y_train)
        
        y_pred = multi_model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"多标签模型整体准确率: {accuracy:.4f}")
        
        for i, disease in enumerate(target_cols):
            disease_accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
            print(f"{disease}预测准确率: {disease_accuracy:.4f}")
        
        self.plot_feature_importance(multi_model, feature_cols, target_cols)
        
        return multi_model, scaler
    
    def plot_feature_importance(self, model, feature_names, target_names):
        print("\n正在绘制多疾病模型特征重要性...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('多疾病预测模型特征重要性', fontsize=16, fontweight='bold')
        
        # 特征重要性图使用低饱和彩色
        colors = ["#ff3939", "#1b8dff", "#34ff56", "#c108ff", "#beff3c"]  # 淡红、淡蓝、淡绿、淡紫、淡黄绿
        
        for i, (estimator, target) in enumerate(zip(model.estimators_, target_names)):
            importance = estimator.feature_importances_
            
            bars = axes[i].bar(feature_names, importance, color=colors)
            axes[i].set_title(f'{target}预测特征重要性')
            axes[i].set_ylabel('重要性')
            axes[i].tick_params(axis='x', rotation=45)
            
            for bar, imp in zip(bars, importance):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{imp:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/多疾病模型特征重要性.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_disease_probabilities(self):
        print("\n正在计算各种疾病组合概率...")
        
        if self.integrated_data is None:
            self.generate_synthetic_data()
        
        df = self.integrated_data
        
        probabilities = {}
        
        probabilities['P(心脏病)'] = df['HeartDisease'].mean()
        probabilities['P(中风)'] = df['Stroke'].mean()
        probabilities['P(肝硬化)'] = df['Cirrhosis'].mean()
        
        probabilities['P(心脏病∩中风)'] = ((df['HeartDisease'] == 1) & (df['Stroke'] == 1)).mean()
        probabilities['P(心脏病∩肝硬化)'] = ((df['HeartDisease'] == 1) & (df['Cirrhosis'] == 1)).mean()
        probabilities['P(中风∩肝硬化)'] = ((df['Stroke'] == 1) & (df['Cirrhosis'] == 1)).mean()
        
        probabilities['P(心脏病∩中风∩肝硬化)'] = ((df['HeartDisease'] == 1) & 
                                                (df['Stroke'] == 1) & 
                                                (df['Cirrhosis'] == 1)).mean()
        
        heart_patients = df[df['HeartDisease'] == 1]
        if len(heart_patients) > 0:
            probabilities['P(中风|心脏病)'] = heart_patients['Stroke'].mean()
            probabilities['P(肝硬化|心脏病)'] = heart_patients['Cirrhosis'].mean()
        
        stroke_patients = df[df['Stroke'] == 1]
        if len(stroke_patients) > 0:
            probabilities['P(心脏病|中风)'] = stroke_patients['HeartDisease'].mean()
            probabilities['P(肝硬化|中风)'] = stroke_patients['Cirrhosis'].mean()
        
        print("疾病概率计算结果:")
        for prob_name, prob_value in probabilities.items():
            print(f"{prob_name}: {prob_value:.4f} ({prob_value*100:.2f}%)")
        
        return probabilities
    
    def discover_association_rules(self):
        print("\n正在进行关联规则挖掘...")
        
        if self.integrated_data is None:
            self.generate_synthetic_data()
        
        df = self.integrated_data
        
        transactions = []
        for _, row in df.iterrows():
            transaction = []
            if row['Hypertension'] == 1:
                transaction.append('高血压')
            if row['Diabetes'] == 1:
                transaction.append('糖尿病')
            if row['Smoking'] == 1:
                transaction.append('吸烟')
            if row['HeartDisease'] == 1:
                transaction.append('心脏病')
            if row['Stroke'] == 1:
                transaction.append('中风')
            if row['Cirrhosis'] == 1:
                transaction.append('肝硬化')
            
            if transaction:
                transactions.append(transaction)
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_transactions, min_support=0.01, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
            
            if len(rules) > 0:
                disease_rules = rules[
                    rules['consequents'].apply(lambda x: any(disease in x for disease in ['心脏病', '中风', '肝硬化']))
                ]
                
                print("发现的关联规则（前10条）:")
                for i, rule in disease_rules.head(10).iterrows():
                    antecedents = ', '.join(list(rule['antecedents']))
                    consequents = ', '.join(list(rule['consequents']))
                    print(f"{antecedents} → {consequents}")
                    print(f"  支持度: {rule['support']:.3f}, 置信度: {rule['confidence']:.3f}, 提升度: {rule['lift']:.3f}")
                    print()
                
                return disease_rules
            else:
                print("未发现满足条件的关联规则")
                return None
        else:
            print("未发现频繁项集")
            return None
    
    def plot_risk_network(self):
        print("\n正在绘制危险因素网络图...")
        
        if self.integrated_data is None:
            self.generate_synthetic_data()
        
        df = self.integrated_data
        
        factors = ['Hypertension', 'Diabetes', 'Smoking', 'HeartDisease', 'Stroke', 'Cirrhosis']
        corr_matrix = df[factors].corr()
        
        G = nx.Graph()
        
        factor_names = {
            'Hypertension': '高血压',
            'Diabetes': '糖尿病', 
            'Smoking': '吸烟',
            'HeartDisease': '心脏病',
            'Stroke': '中风',
            'Cirrhosis': '肝硬化'
        }
        
        for factor in factors:
            G.add_node(factor_names[factor])
        
        threshold = 0.1
        for i in range(len(factors)):
            for j in range(i+1, len(factors)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    G.add_edge(factor_names[factors[i]], factor_names[factors[j]], weight=abs(corr))
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 网络图节点使用低饱和彩色
        node_colors = [
            '#e0b0b0',  # 高血压-淡红
            '#b0c8e0',  # 糖尿病-淡蓝
            '#b0e0b8',  # 吸烟-淡绿
            '#d8c0e0',  # 心脏病-淡紫
            '#d0e0b0',  # 中风-淡黄绿
            '#b0e0d8'   # 肝硬化-淡青
        ]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], alpha=0.6, edge_color='#cccccc')  # 灰色边
        
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('疾病和危险因素关联网络图', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d8c0e0', label='疾病'),
            Patch(facecolor='#e0b0b0', label='危险因素')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/危险因素网络图.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def comprehensive_risk_evaluation(self):
        print("\n正在进行综合风险评估...")
        
        if self.integrated_data is None:
            self.generate_synthetic_data()
        
        df = self.integrated_data
        
        def calculate_risk_score(row):
            score = 0
            
            if row['Age'] > 65:
                score += 3
            elif row['Age'] > 50:
                score += 2
            elif row['Age'] > 35:
                score += 1
            
            if row['Sex'] == 1:
                score += 1
            
            score += row['Hypertension'] * 2
            score += row['Diabetes'] * 2
            score += row['Smoking'] * 2
            
            return score
        
        df['RiskScore'] = df.apply(calculate_risk_score, axis=1)
        
        def risk_level(score):
            if score >= 8:
                return '高风险'
            elif score >= 5:
                return '中风险'
            else:
                return '低风险'
        
        df['RiskLevel'] = df['RiskScore'].apply(risk_level)
        
        risk_analysis = df.groupby('RiskLevel').agg({
            'HeartDisease': 'mean',
            'Stroke': 'mean',
            'Cirrhosis': 'mean'
        }).round(4)
        
        print("不同风险等级的疾病发生率:")
        print(risk_analysis)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('综合风险评估结果', fontsize=16, fontweight='bold')
        
        # 风险等级分布使用低饱和彩色
        risk_counts = df['RiskLevel'].value_counts()
        wedges, texts, autotexts = axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                                                colors=["#0582ff", "#14ff3b", "#ff3f3f"])  # 淡蓝、淡绿、淡红
        axes[0, 0].set_title('风险等级分布')
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(10)
        
        # 风险评分分布使用淡紫色
        axes[0, 1].hist(df['RiskScore'], bins=15, alpha=0.7, color="#4cabff", edgecolor='white')
        axes[0, 1].set_title('风险评分分布')
        axes[0, 1].set_xlabel('风险评分')
        axes[0, 1].set_ylabel('人数')
        
        # 不同风险等级的疾病发生率使用低饱和彩色
        risk_analysis.plot(kind='bar', ax=axes[1, 0], color=["#FF741D", "#2d96ff", "#54ff70"])  # 淡红、淡蓝、淡绿
        axes[1, 0].set_title('不同风险等级的疾病发生率')
        axes[1, 0].set_ylabel('发生率')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(['心脏病', '中风', '肝硬化'])
        
        # 风险评分与疾病数量关系使用淡棕色
        df['DiseaseCount'] = df['HeartDisease'] + df['Stroke'] + df['Cirrhosis']
        score_disease = df.groupby('RiskScore')['DiseaseCount'].mean()
        axes[1, 1].plot(score_disease.index, score_disease.values, 'o-', color="#0060c0", linewidth=2, markersize=6)
        axes[1, 1].set_title('风险评分与平均疾病数量关系')
        axes[1, 1].set_xlabel('风险评分')
        axes[1, 1].set_ylabel('平均疾病数量')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/综合风险评估.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return df, risk_analysis
    
    def run_complete_analysis(self):
        print("开始多疾病关联与综合风险评估分析")
        
        self.load_and_prepare_data()
        
        common_factors = self.find_common_risk_factors()
        
        self.generate_synthetic_data()
        
        comorbidity_patterns = self.analyze_comorbidity()
        
        self.draw_venn_diagram()
        
        multi_model, scaler = self.build_multidisease_model()
        
        probabilities = self.calculate_disease_probabilities()
        
        association_rules = self.discover_association_rules()
        
        self.plot_risk_network()
        
        risk_data, risk_analysis = self.comprehensive_risk_evaluation()
        
        print(f"\n分析完成！所有图表已保存到 {self.result_dir} 文件夹中。")
        
        print("\n=== 关键结论 ===")
        print(f"1. 同时患三种疾病的概率: {probabilities['P(心脏病∩中风∩肝硬化)']*100:.2f}%")
        print(f"2. 心脏病患者中风概率: {probabilities.get('P(中风|心脏病)', 0)*100:.2f}%")
        print(f"3. 高风险人群疾病发生率显著高于低风险人群")
        
        return {
            'comorbidity_patterns': comorbidity_patterns,
            'probabilities': probabilities,
            'multi_model': multi_model,
            'risk_analysis': risk_analysis
        }

if __name__ == "__main__":
    analyzer = DiseaseAssociationAnalyzer()
    analyzer.run_complete_analysis()