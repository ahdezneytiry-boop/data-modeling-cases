# -*- coding: utf-8 -*-
"""
问题4：预防三种疾病的建议和措施（修改版：调整颜色、注释及添加词云）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud, STOPWORDS
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（确保词云中文正常显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class Problem4PreventionRecommendations:
    def __init__(self):
        self.output_dir = "问题4_预防建议和措施（修改版）"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.prevention_strategies = {}
        self.who_letter = ""
        
    def analyze_prevention_strategies(self):
        """分析预防策略"""
        print("正在分析疾病预防策略...")
        
        self.prevention_strategies = {
            '心脏病预防': {
                '早期筛查': [
                    '40岁以上人群每年进行心电图检查',
                    '高血压、糖尿病患者每6个月检查一次',
                    '有家族史人群35岁开始定期筛查',
                    '血脂、血糖、血压三项联合筛查'
                ],
                '生活方式干预': [
                    '控制血压：目标<140/90mmHg',
                    '控制血糖：糖化血红蛋白<7%',
                    '控制血脂：LDL-C<2.6mmol/L',
                    '戒烟限酒：完全戒烟，限制饮酒',
                    '合理饮食：地中海饮食模式',
                    '规律运动：每周150分钟中等强度运动'
                ],
                '药物预防': [
                    '阿司匹林一级预防（高危人群）',
                    '他汀类药物降脂治疗',
                    'ACEI/ARB类药物控制血压',
                    '二甲双胍控制血糖'
                ]
            },
            '中风预防': {
                '早期筛查': [
                    '颈动脉超声检查（50岁以上）',
                    '血压监测（每日家庭监测）',
                    '心房颤动筛查（65岁以上）',
                    '血管危险因素评估'
                ],
                '生活方式干预': [
                    '严格控制血压：目标<130/80mmHg',
                    '房颤患者抗凝治疗',
                    '控制血糖和血脂',
                    '戒烟：降低中风风险50%',
                    '限盐饮食：每日<6g',
                    '增加钾摄入：新鲜蔬果'
                ],
                '药物预防': [
                    '抗血小板聚集药物',
                    '抗凝药物（房颤患者）',
                    '降压药物规范使用',
                    '他汀类药物稳定斑块'
                ]
            },
            '肝硬化预防': {
                '早期筛查': [
                    '肝功能检查（每年）',
                    '乙肝、丙肝病毒筛查',
                    '肝脏超声检查',
                    '肝纤维化指标检测'
                ],
                '生活方式干预': [
                    '限制饮酒：男性<25g/日，女性<15g/日',
                    '控制体重：BMI 18.5-24.9',
                    '疫苗接种：乙肝疫苗',
                    '避免肝毒性药物',
                    '健康饮食：低脂低糖',
                    '规律作息：充足睡眠'
                ],
                '药物预防': [
                    '抗病毒治疗（乙肝、丙肝）',
                    '保肝药物使用',
                    '抗纤维化治疗',
                    '并发症预防用药'
                ]
            }
        }
        
        return self.prevention_strategies
    
    def create_wordcloud(self):
        """生成预防措施关键词词云图"""
        print("正在生成预防措施词云图...")
        
        # 提取所有预防措施文本
        text = ""
        for disease in self.prevention_strategies.values():
            for category in disease.values():
                text += " ".join(category) + " "
        
        # 配置词云（支持中文，调整颜色）
        wc = WordCloud(
            font_path="simhei.ttf",  # 确保有中文字体
            width=1200,
            height=800,
            background_color='white',
            colormap='viridis',  # 调整颜色方案
            max_words=100,
            stopwords=STOPWORDS.update(['患者', '每年', '每日']),  # 过滤无意义词
            contour_width=1,
            contour_color='steelblue'
        )
        
        # 生成词云
        wc.generate(text)
        
        # 保存词云图
        plt.figure(figsize=(12, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('疾病预防措施关键词词云', fontsize=16, pad=20)
        plt.savefig(f"{self.output_dir}/问题4_预防措施词云.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return text
    
    def create_prevention_effectiveness_analysis(self):
        """创建预防措施有效性分析（调整颜色和注释）"""
        print("正在分析预防措施有效性...")
        
        prevention_data = {
            '预防措施': [
                '戒烟', '控制血压', '控制血糖', '规律运动', '健康饮食',
                '限制饮酒', '体重控制', '定期筛查', '药物预防', '疫苗接种'
            ],
            '心脏病风险降低(%)': [50, 40, 35, 30, 25, 15, 20, 60, 45, 5],
            '中风风险降低(%)': [50, 45, 30, 25, 20, 20, 15, 55, 40, 5],
            '肝硬化风险降低(%)': [30, 15, 20, 25, 30, 70, 35, 80, 60, 90],
            '实施难度(1-5)': [4, 3, 3, 3, 2, 4, 3, 2, 2, 1],
            '成本效益比': [9.2, 8.5, 7.8, 7.2, 6.8, 8.8, 6.5, 9.5, 7.0, 9.8]
        }
        
        df = pd.DataFrame(prevention_data)
        
        # 绘制预防措施有效性对比图（调整颜色）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('疾病预防措施有效性分析', fontsize=16, fontweight='bold')
        
        # 1. 风险降低效果对比（修改颜色为seaborn调色板）
        diseases = ['心脏病风险降低(%)', '中风风险降低(%)', '肝硬化风险降低(%)']
        x = np.arange(len(df))
        width = 0.25
        colors = sns.color_palette('husl', 3)  # 更和谐的颜色方案
        
        for i, (disease, color) in enumerate(zip(diseases, colors)):
            axes[0, 0].bar(x + i*width, df[disease], width, 
                          label=disease.replace('风险降低(%)', ''), 
                          alpha=0.8, color=color)
        
        axes[0, 0].set_xlabel('预防措施', fontsize=10)
        axes[0, 0].set_ylabel('风险降低百分比', fontsize=10)
        axes[0, 0].set_title('各预防措施对三种疾病的风险降低效果', fontsize=12)
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(df['预防措施'], rotation=45, ha='right', fontsize=9)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成本效益分析（修改颜色和注释位置）
        scatter = axes[0, 1].scatter(
            df['实施难度(1-5)'], df['成本效益比'], 
            s=df['心脏病风险降低(%)']*3, 
            alpha=0.6, 
            c=df['综合效果'] if '综合效果' in df.columns else df.index,  # 按综合效果着色
            cmap='coolwarm'  # 新的颜色映射
        )
        axes[0, 1].set_xlabel('实施难度 (1=容易, 5=困难)', fontsize=10)
        axes[0, 1].set_ylabel('成本效益比', fontsize=10)
        axes[0, 1].set_title('预防措施成本效益分析（气泡大小：心脏病风险降低程度）', fontsize=12)
        
        # 优化注释：调整位置避免重叠
        for i, txt in enumerate(df['预防措施']):
            xytext = (5, 5) if i % 2 == 0 else (5, -15)  # 交替位置
            axes[0, 1].annotate(txt, 
                              (df['实施难度(1-5)'].iloc[i], df['成本效益比'].iloc[i]),
                              xytext=xytext, textcoords='offset points', 
                              fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        
        # 3. 综合效果排名（修改颜色）
        df['综合效果'] = (df['心脏病风险降低(%)'] + df['中风风险降低(%)'] + df['肝硬化风险降低(%)']) / 3
        df_sorted = df.sort_values('综合效果', ascending=True)
        
        bars = axes[1, 0].barh(
            df_sorted['预防措施'], 
            df_sorted['综合效果'], 
            color=sns.color_palette('Greens', len(df_sorted))  # 渐变色
        )
        axes[1, 0].set_xlabel('平均风险降低百分比', fontsize=10)
        axes[1, 0].set_title('预防措施综合效果排名', fontsize=12)
        
        # 添加数值标签（更清晰的格式）
        for bar, value in zip(bars, df_sorted['综合效果']):
            axes[1, 0].text(
                bar.get_width() + 1, 
                bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', 
                va='center', 
                fontsize=9, 
                fontweight='bold'
            )
        
        # 4. 实施优先级矩阵（修改颜色和注释）
        scatter = axes[1, 1].scatter(
            df['综合效果'], df['成本效益比'], 
            s=100, 
            alpha=0.7, 
            c=df['实施难度(1-5)'],  # 按难度着色
            cmap='plasma'  # 新的颜色映射
        )
        
        # 添加颜色条说明难度
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('实施难度(1-5)', fontsize=9)
        
        # 象限线和标签（优化样式）
        axes[1, 1].axhline(y=df['成本效益比'].median(), color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=df['综合效果'].median(), color='gray', linestyle='--', alpha=0.5)
        
        axes[1, 1].set_xlabel('综合预防效果', fontsize=10)
        axes[1, 1].set_ylabel('成本效益比', fontsize=10)
        axes[1, 1].set_title('预防措施优先级矩阵', fontsize=12)
        
        # 象限标签（更醒目的样式）
        quad_labels = [
            ('高效果\n高效益', 0.95, 0.95, 'lightgreen'),
            ('低效果\n高效益', 0.05, 0.95, 'yellow'),
            ('高效果\n低效益', 0.95, 0.05, 'orange'),
            ('低效果\n低效益', 0.05, 0.05, 'lightcoral')
        ]
        for label, x, y, color in quad_labels:
            axes[1, 1].text(
                x, y, label, 
                transform=axes[1, 1].transAxes,
                ha='right' if x > 0.5 else 'left', 
                va='top' if y > 0.5 else 'bottom', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.5),
                fontsize=9
            )
        
        # 措施标签（避免重叠）
        for i, txt in enumerate(df['预防措施']):
            axes[1, 1].annotate(
                txt, 
                (df['综合效果'].iloc[i], df['成本效益比'].iloc[i]),
                xytext=(5, -10) if i % 2 == 0 else (-40, 5),  # 交替位置
                textcoords='offset points', 
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6)
            )
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/问题4_预防措施有效性分析.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
    
    def create_screening_strategy_flowchart(self):
        """创建筛查策略流程图（调整颜色）"""
        print("正在创建筛查策略流程图...")
        
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
        
        df_screening = pd.DataFrame(screening_data)
        
        # 绘制筛查策略表格（修改颜色）
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格（使用新的颜色方案）
        table = ax.table(
            cellText=df_screening.values,
            colLabels=df_screening.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 表头颜色修改为深蓝色
        header_color = '#2c3e50'
        for i in range(len(df_screening.columns)):
            table[(0, i)].set_facecolor(header_color)
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        
        # 表格内容行颜色交替（浅灰/白色）
        row_colors = ['#f8f9fa', '#e9ecef']
        for i in range(1, len(df_screening) + 1):
            for j in range(len(df_screening.columns)):
                table[(i, j)].set_facecolor(row_colors[i % 2])
                table[(i, j)].set_text_props(fontsize=10)
        
        plt.title('分年龄段疾病筛查策略', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f"{self.output_dir}/问题4_筛查策略表.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return df_screening
    
    def create_lifestyle_intervention_guide(self):
        """创建生活方式干预指南（调整颜色）"""
        print("正在创建生活方式干预指南...")
        
        lifestyle_data = {
            '干预类型': ['饮食控制', '运动锻炼', '戒烟限酒', '压力管理', '睡眠质量'],
            '具体措施': [
                '地中海饮食\n减少饱和脂肪\n增加纤维摄入\n控制钠盐',
                '有氧运动150分钟/周\n力量训练2次/周\n日常活动增加\n循序渐进',
                '完全戒烟\n限制饮酒\n避免二手烟\n寻求专业帮助',
                '冥想放松\n规律作息\n社交支持\n心理咨询',
                '7-9小时睡眠\n规律睡眠时间\n良好睡眠环境\n避免睡前刺激'
            ],
            '预期效果': [
                '降低心血管风险25%\n改善血脂水平\n控制体重',
                '降低死亡率30%\n改善心肺功能\n增强免疫力',
                '降低中风风险50%\n减少癌症风险\n改善肺功能',
                '降低血压\n改善心理健康\n提高生活质量',
                '改善代谢\n增强免疫\n提高认知功能'
            ],
            '实施难度': [3, 3, 5, 4, 2]
        }
        
        # 绘制生活方式干预雷达图（调整颜色）
        categories = lifestyle_data['干预类型']
        effectiveness = [25, 30, 50, 20, 15]  # 风险降低百分比
        feasibility = [7, 7, 3, 6, 8]  # 可行性评分(1-10)
        
        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        effectiveness += effectiveness[:1]
        feasibility += feasibility[:1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
        
        # 1. 有效性雷达图（修改颜色）
        ax1.plot(angles, effectiveness, 'o-', linewidth=2, label='风险降低效果', color='#e74c3c')
        ax1.fill(angles, effectiveness, alpha=0.25, color='#e74c3c')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, fontsize=9)
        ax1.set_ylim(0, 60)
        ax1.set_title('生活方式干预有效性', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, color='#ecf0f1')
        
        # 2. 可行性雷达图（修改颜色）
        ax2.plot(angles, feasibility, 'o-', linewidth=2, label='实施可行性', color='#3498db')
        ax2.fill(angles, feasibility, alpha=0.25, color='#3498db')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontsize=9)
        ax2.set_ylim(0, 10)
        ax2.set_title('生活方式干预可行性', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, color='#ecf0f1')
        
        # 添加图例说明
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/问题4_生活方式干预指南.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return lifestyle_data
    
    def create_public_health_policy_recommendations(self):
        """创建公共卫生政策建议（调整颜色）"""
        print("正在创建公共卫生政策建议...")
        
        policy_data = {
            '政策类型': [
                '税收政策', '环境政策', '教育政策', '医疗政策', '监管政策'
            ],
            '具体措施': [
                '高盐高糖食品征税\n烟草税提高\n健康食品补贴',
                '无烟环境建设\n运动设施增加\n空气质量改善',
                '健康教育普及\n营养标签强制\n媒体宣传',
                '基本医疗保障\n预防服务免费\n分级诊疗',
                '食品安全监管\n药品质量控制\n医疗质量标准'
            ],
            '预期影响': [
                '减少不健康食品消费\n增加政府收入\n改善人群健康',
                '降低环境风险因素\n促进健康行为\n提高生活质量',
                '提高健康意识\n改变行为习惯\n预防疾病发生',
                '提高医疗可及性\n早期发现疾病\n降低医疗费用',
                '保障食品药品安全\n规范医疗行为\n保护患者权益'
            ],
            '实施成本': ['中等', '高', '低', '高', '中等'],
            '预期效果': ['高', '中等', '中等', '高', '中等']
        }
        
        # 绘制政策建议矩阵（调整颜色）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('公共卫生政策建议分析', fontsize=16, fontweight='bold')
        
        # 1. 政策类型分布（修改颜色）
        policy_counts = pd.Series(policy_data['政策类型']).value_counts()
        axes[0, 0].pie(
            [1]*len(policy_data['政策类型']), 
            labels=policy_data['政策类型'], 
            autopct='%1.1f%%', 
            startangle=90,
            colors=sns.color_palette('pastel')  # 柔和的颜色
        )
        axes[0, 0].set_title('政策类型分布', fontsize=12)
        
        # 2. 成本效果矩阵（修改颜色）
        cost_map = {'低': 1, '中等': 2, '高': 3}
        effect_map = {'低': 1, '中等': 2, '高': 3}
        
        costs = [cost_map[cost] for cost in policy_data['实施成本']]
        effects = [effect_map[effect] for effect in policy_data['预期效果']]
        
        scatter = axes[0, 1].scatter(
            costs, effects, 
            s=200, 
            alpha=0.7, 
            c=range(len(costs)), 
            cmap='Set2'  # 新的颜色映射
        )
        axes[0, 1].set_xlabel('实施成本', fontsize=10)
        axes[0, 1].set_ylabel('预期效果', fontsize=10)
        axes[0, 1].set_title('政策成本效果分析', fontsize=12)
        axes[0, 1].set_xticks([1, 2, 3])
        axes[0, 1].set_xticklabels(['低', '中等', '高'])
        axes[0, 1].set_yticks([1, 2, 3])
        axes[0, 1].set_yticklabels(['低', '中等', '高'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加政策标签（优化位置）
        for i, policy in enumerate(policy_data['政策类型']):
            axes[0, 1].annotate(
                policy, 
                (costs[i], effects[i]),
                xytext=(5, 5) if i < 3 else (-30, -15),  # 分组调整位置
                textcoords='offset points', 
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6)
            )
        
        # 3. 政策优先级排序（修改颜色）
        priority_scores = []
        for i in range(len(policy_data['政策类型'])):
            score = effects[i] - costs[i] + 3  # 优先级分数
            priority_scores.append(score)
        
        policy_priority = pd.DataFrame({
            '政策': policy_data['政策类型'],
            '优先级分数': priority_scores
        }).sort_values('优先级分数', ascending=True)
        
        bars = axes[1, 0].barh(
            policy_priority['政策'], 
            policy_priority['优先级分数'], 
            color=sns.color_palette('Blues_r', len(policy_priority))  # 逆向渐变
        )
        axes[1, 0].set_xlabel('优先级分数', fontsize=10)
        axes[1, 0].set_title('政策实施优先级排序', fontsize=12)
        
        # 添加分数标签
        for bar, score in zip(bars, policy_priority['优先级分数']):
            axes[1, 0].text(
                bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,
                f'{score}', 
                va='center',
                fontsize=9,
                fontweight='bold'
            )
        
        # 4. 政策实施时间线（修改颜色）
        timeline_data = {
            '短期(1年内)': ['健康教育普及', '营养标签强制'],
            '中期(1-3年)': ['无烟环境建设', '预防服务免费'],
            '长期(3-5年)': ['高盐高糖食品征税', '分级诊疗体系']
        }
        
        y_pos = 0
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # 更鲜明的颜色
        
        for i, (period, policies) in enumerate(timeline_data.items()):
            axes[1, 1].barh(
                [y_pos + j*0.3 for j in range(len(policies))], 
                [1]*len(policies), 
                height=0.25, 
                color=colors[i], 
                alpha=0.7, 
                label=period
            )
            
            for j, policy in enumerate(policies):
                axes[1, 1].text(
                    0.5, 
                    y_pos + j*0.3, 
                    policy, 
                    va='center', 
                    ha='center', 
                    fontsize=8,
                    fontweight='bold'
                )
            
            y_pos += len(policies) * 0.3 + 0.2
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(-0.2, y_pos)
        axes[1, 1].set_xlabel('实施进度', fontsize=10)
        axes[1, 1].set_title('政策实施时间线', fontsize=12)
        axes[1, 1].legend(loc='upper right', fontsize=9)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/问题4_公共卫生政策建议.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return policy_data
    
    def generate_who_letter(self):
        """生成给WHO的建议信"""
        print("正在生成给WHO的建议信...")
        
        self.who_letter = """
致世界卫生组织的建议信

尊敬的世界卫生组织领导：

基于我们对心血管疾病、中风和肝硬化的综合分析，我们提出以下预防策略和政策建议：

一、早期筛查策略建议

1. 建立分层筛查体系
   - 高风险人群（45岁以上，有危险因素）：每年筛查
   - 中等风险人群（35-45岁）：每2-3年筛查
   - 低风险人群（35岁以下）：每5年筛查

2. 重点筛查项目
   - 血压、血糖、血脂三项联合检测
   - 心电图和颈动脉超声检查
   - 肝功能和病毒性肝炎筛查
   - 心房颤动筛查（65岁以上人群）

二、生活方式干预建议

1. 控制主要危险因素
   - 血压控制：目标<130/80mmHg
   - 血糖控制：糖化血红蛋白<7%
   - 血脂控制：LDL-C<2.6mmol/L
   - 体重控制：BMI 18.5-24.9 kg/m²

2. 推广健康生活方式
   - 戒烟：可降低心血管疾病风险50%
   - 限酒：男性<25g/日，女性<15g/日
   - 运动：每周150分钟中等强度有氧运动
   - 饮食：推广地中海饮食模式

三、公共卫生政策建议

1. 税收政策
   - 对高盐、高糖、高脂食品征收健康税
   - 提高烟草税收，降低吸烟率
   - 对健康食品提供补贴

2. 环境政策
   - 建设无烟环境，保护公众免受二手烟危害
   - 增加公共运动设施，促进体育锻炼
   - 改善空气质量，减少环境污染

3. 教育政策
   - 在学校开展健康教育课程
   - 强制要求食品营养标签
   - 利用媒体开展健康宣传活动

4. 医疗政策
   - 将预防服务纳入基本医疗保障
   - 建立完善的分级诊疗体系
   - 加强基层医疗机构能力建设

四、实施建议

1. 短期目标（1年内）
   - 启动全球健康教育计划
   - 建立疾病监测网络
   - 制定统一的筛查标准

2. 中期目标（1-3年）
   - 实施税收政策改革
   - 建设无烟环境
   - 推广预防服务

3. 长期目标（3-5年）
   - 建立全球疾病预防体系
   - 实现可持续发展目标
   - 显著降低疾病负担

五、预期效果

通过实施上述建议，预计可以：
- 降低心血管疾病发病率25-30%
- 减少中风发生率20-25%
- 降低肝硬化死亡率30-40%
- 节约全球医疗费用数千亿美元

我们相信，通过全球合作和共同努力，可以有效预防和控制这三种主要疾病，改善全人类的健康水平。

此致
敬礼！

疾病预防与大数据分析研究团队
2025年
        """
        
        # 保存建议信
        with open(f"{self.output_dir}/问题4_WHO建议信.txt", 'w', encoding='utf-8') as f:
            f.write(self.who_letter)
        
        print("WHO建议信已生成并保存")
        return self.who_letter
    
    def create_prevention_cost_benefit_analysis(self):
        """创建预防措施成本效益分析（调整颜色）"""
        print("正在创建预防措施成本效益分析...")
        
        cost_benefit_data = {
            '预防措施': [
                '全民健康教育', '高危人群筛查', '戒烟项目', 
                '血压控制项目', '糖尿病管理', '疫苗接种'
            ],
            '实施成本(万元)': [500, 2000, 1500, 3000, 2500, 800],
            '预防疾病数': [1200, 800, 1500, 2000, 1000, 300],
            '节约医疗费用(万元)': [6000, 8000, 15000, 20000, 12000, 3000],
            '净收益(万元)': [5500, 6000, 13500, 17000, 9500, 2200],
            '投资回报率': [11, 4, 9, 6.7, 4.8, 3.75]
        }
        
        df_cost = pd.DataFrame(cost_benefit_data)
        
        # 绘制成本效益分析图（调整颜色）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('疾病预防措施成本效益分析', fontsize=16, fontweight='bold')
        palette = sns.color_palette('muted')  # 统一的柔和色调
        
        # 1. 投资回报率对比
        bars1 = axes[0, 0].bar(
            df_cost['预防措施'], 
            df_cost['投资回报率'], 
            color=palette[0], 
            alpha=0.8
        )
        axes[0, 0].set_ylabel('投资回报率', fontsize=10)
        axes[0, 0].set_title('各预防措施投资回报率', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars1, df_cost['投资回报率']):
            axes[0, 0].text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.1,
                f'{value:.1f}', 
                ha='center', 
                va='bottom',
                fontsize=9
            )
        
        # 2. 成本与收益对比
        x = np.arange(len(df_cost))
        width = 0.35
        
        bars2 = axes[0, 1].bar(
            x - width/2, 
            df_cost['实施成本(万元)'], 
            width, 
            label='实施成本', 
            color=palette[1], 
            alpha=0.8
        )
        bars3 = axes[0, 1].bar(
            x + width/2, 
            df_cost['节约医疗费用(万元)'], 
            width,
            label='节约费用', 
            color=palette[2], 
            alpha=0.8
        )
        
        axes[0, 1].set_ylabel('费用(万元)', fontsize=10)
        axes[0, 1].set_title('预防措施成本与收益对比', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df_cost['预防措施'], rotation=45, ha='right', fontsize=9)
        axes[0, 1].legend(fontsize=9)
        
        # 3. 预防疾病数量
        bars4 = axes[1, 0].bar(
            df_cost['预防措施'], 
            df_cost['预防疾病数'], 
            color=palette[3], 
            alpha=0.8
        )
        axes[1, 0].set_ylabel('预防疾病数量', fontsize=10)
        axes[1, 0].set_title('各预防措施预防疾病数量', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars4, df_cost['预防疾病数']):
            axes[1, 0].text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 20,
                f'{value}', 
                ha='center', 
                va='bottom',
                fontsize=9
            )
        
        # 4. 净收益分析
        bars5 = axes[1, 1].bar(
            df_cost['预防措施'], 
            df_cost['净收益(万元)'], 
            color=palette[4], 
            alpha=0.8
        )
        axes[1, 1].set_ylabel('净收益(万元)', fontsize=10)
        axes[1, 1].set_title('各预防措施净收益', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars5, df_cost['净收益(万元)']):
            axes[1, 1].text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 200,
                f'{value}', 
                ha='center', 
                va='bottom',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/问题4_预防措施成本效益分析.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return df_cost
    
    def create_implementation_roadmap(self):
        """创建实施路线图（调整颜色）"""
        print("正在创建实施路线图...")
        
        roadmap_data = {
            '阶段': ['准备阶段', '试点阶段', '推广阶段', '完善阶段', '评估阶段'],
            '时间': ['0-6个月', '6-18个月', '18-36个月', '36-48个月', '48-60个月'],
            '主要任务': [
                '政策制定\n资源准备\n人员培训',
                '试点实施\n效果评估\n方案优化',
                '全面推广\n监督指导\n质量控制',
                '系统完善\n标准化\n可持续发展',
                '效果评估\n经验总结\n国际推广'
            ],
            '预期成果': [
                '完成政策框架\n建立组织体系',
                '验证可行性\n形成标准模式',
                '覆盖目标人群\n初见成效',
                '建立长效机制\n持续改进',
                '总结最佳实践\n国际分享'
            ],
            '关键指标': [
                '政策完成率100%',
                '试点覆盖率20%',
                '全面覆盖率80%',
                '质量达标率95%',
                '目标实现率90%'
            ]
        }
        
        # 绘制实施路线图（修改颜色）
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 使用渐变色表示阶段进展
        colors = sns.color_palette('viridis', len(roadmap_data['阶段']))
        
        # 绘制时间线
        y_pos = 0
        for i, (stage, time, task, outcome, indicator) in enumerate(zip(
            roadmap_data['阶段'], roadmap_data['时间'], roadmap_data['主要任务'],
            roadmap_data['预期成果'], roadmap_data['关键指标']
        )):
            # 阶段框（使用渐变色）
            rect = plt.Rectangle(
                (i*2, 0), 1.8, 4, 
                facecolor=colors[i], 
                alpha=0.8, 
                edgecolor='white', 
                linewidth=2
            )
            ax.add_patch(rect)
            
            # 阶段名称（加粗）
            ax.text(i*2 + 0.9, 3.5, stage, 
                   ha='center', va='center', 
                   fontweight='bold', fontsize=12,
                   color='white' if i > 1 else 'black')  # 深色背景用白色文字
            
            # 时间
            ax.text(i*2 + 0.9, 3, time, 
                   ha='center', va='center', 
                   fontsize=10,
                   color='white' if i > 1 else 'black')
            
            # 主要任务
            ax.text(i*2 + 0.9, 2.2, task, 
                   ha='center', va='center', 
                   fontsize=9,
                   color='white' if i > 1 else 'black')
            
            # 预期成果
            ax.text(i*2 + 0.9, 1.4, outcome, 
                   ha='center', va='center', 
                   fontsize=9,
                   color='white' if i > 1 else 'black')
            
            # 关键指标（突出显示）
            ax.text(i*2 + 0.9, 0.6, indicator, 
                   ha='center', va='center', 
                   fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 连接箭头（更明显）
            if i < len(roadmap_data['阶段']) - 1:
                ax.arrow(
                    i*2 + 1.8, 2, 0.15, 0, 
                    head_width=0.2, head_length=0.1, 
                    fc='black', ec='black', 
                    linewidth=1.5
                )
        
        ax.set_xlim(-0.2, len(roadmap_data['阶段'])*2)
        ax.set_ylim(0, 4)
        ax.set_title('疾病预防措施实施路线图', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/问题4_实施路线图.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return roadmap_data
    
    def run_analysis(self):
        """运行完整分析（新增词云调用）"""
        print("开始问题4：预防三种疾病的建议和措施（修改版）")
        
        # 分析预防策略
        prevention_strategies = self.analyze_prevention_strategies()
        
        # 生成词云图（新增步骤）
        self.create_wordcloud()
        
        # 创建预防措施有效性分析
        effectiveness_data = self.create_prevention_effectiveness_analysis()
        
        # 创建筛查策略
        screening_strategy = self.create_screening_strategy_flowchart()
        
        # 创建生活方式干预指南
        lifestyle_guide = self.create_lifestyle_intervention_guide()
        
        # 创建公共卫生政策建议
        policy_recommendations = self.create_public_health_policy_recommendations()
        
        # 生成WHO建议信
        who_letter = self.generate_who_letter()
        
        # 创建成本效益分析
        cost_benefit_analysis = self.create_prevention_cost_benefit_analysis()
        
        # 创建实施路线图
        implementation_roadmap = self.create_implementation_roadmap()
        
        print(f"\n问题4分析完成！所有图表和建议信已保存到 {self.output_dir} 文件夹中。")
        print("新增内容：预防措施关键词词云图")
        
        # 输出关键建议
        print("\n=== 关键预防建议 ===")
        print("1. 建立分层筛查体系，重点关注高危人群")
        print("2. 推广生活方式干预，戒烟限酒效果最显著")
        print("3. 实施公共卫生政策，税收和环境政策优先")
        print("4. 加强国际合作，共同应对全球健康挑战")
        
        return {
            'prevention_strategies': prevention_strategies,
            'effectiveness_data': effectiveness_data,
            'policy_recommendations': policy_recommendations,
            'who_letter': who_letter,
            'cost_benefit_analysis': cost_benefit_analysis
        }

if __name__ == "__main__":
    problem4 = Problem4PreventionRecommendations()
    problem4.run_analysis()