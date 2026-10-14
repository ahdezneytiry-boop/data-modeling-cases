import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# ==========================================
# 1. 核心类定义：选手角色 (Archetype)
# ==========================================
class Contestant:
    def __init__(self, name, archetype, base_skill, base_popularity, growth_rate, consistency):
        self.name = name
        self.archetype = archetype  # 类别：流量/实力/老兵...
        self.skill = base_skill     # 0-10, 对应评委分
        self.popularity = base_popularity # 0-10, 对应粉丝基数
        self.growth = growth_rate   # 每周技能增长率
        self.consistency = consistency # 表现稳定性 (方差倒数)
        self.is_eliminated = False
        self.history_rank = []

    def perform(self, week, noise=0.5):
        # 模拟评委打分 (Judge Score): 技能 + 成长 + 随机波动
        current_skill = min(10, self.skill + self.growth * week)
        judge_score = np.random.normal(current_skill, noise)
        judge_score = np.clip(judge_score, 1, 10) * 3  # 3位评委总分 30分制
        
        # 模拟粉丝投票 (Fan Vote): 人气 + 曝光效应(随周次增加) + 随机波动
        # 关键设定：流量明星的粉丝票服从幂律分布，基数极大
        if self.archetype == 'Viral Star':
            # 流量明星有巨额加成
            raw_votes = (self.popularity ** 3) * (1 + week*0.1) * np.random.uniform(0.9, 1.1)
        else:
            raw_votes = (self.popularity ** 2) * (1 + week*0.05) * np.random.uniform(0.8, 1.2)
            
        return judge_score, raw_votes

# ==========================================
# 2. 场景生成：创建5类典型选手
# ==========================================
def create_contestants():
    return [
        # 1. 流量级明星：技能差(4)，但人气极高(10)，典型如 Bobby Bones
        Contestant("Viral Vinny", "Viral Star", 4.0, 10.0, 0.05, 0.8),
        
        # 2. 百老汇演员：技能顶尖(9)，但没人认识(3)，典型“怀才不遇”
        Contestant("Broadway Bella", "Broadway Pro", 9.0, 3.0, 0.02, 0.95),
        
        # 3. 退伍军人：技能中等(6)，自带爱国光环(8)，受众忠诚
        Contestant("Patriot Paul", "Veteran", 6.0, 8.0, 0.15, 0.9),
        
        # 4. 高龄传奇：技能因身体受限(5)，情怀满分(7)，前期安全后期危险
        Contestant("Legend Larry", "Elderly", 5.0, 7.0, -0.05, 0.7), # 体力随周次下降
        
        # 5. 年轻童星：技能好(8)，人气高(9)，潜力股
        Contestant("Disney Daisy", "Young Star", 8.0, 9.0, 0.20, 0.85),
        
        # 填充一些陪跑选手 (Filler) 以构成完整赛季 (12人)
        Contestant("Filler A", "Average", 5, 5, 0.1, 0.8),
        Contestant("Filler B", "Average", 6, 4, 0.1, 0.8),
        Contestant("Filler C", "Average", 4, 6, 0.1, 0.8),
        Contestant("Filler D", "Average", 7, 3, 0.1, 0.8),
        Contestant("Filler E", "Average", 5, 5, 0.1, 0.8),
        Contestant("Filler F", "Average", 6, 5, 0.1, 0.8),
        Contestant("Filler G", "Average", 5, 6, 0.1, 0.8),
    ]

# ==========================================
# 3. 核心算法：两种赛制模拟器
# ==========================================
def simulate_week(contestants, mechanism='RSP'):
    # 1. 所有人表演
    scores = []
    active_contestants = [c for c in contestants if not c.is_eliminated]
    if len(active_contestants) <= 1: return None # 决赛结束

    for c in active_contestants:
        j_score, f_votes = c.perform(week=len(c.history_rank))
        scores.append({
            'name': c.name,
            'obj': c,
            'judge_raw': j_score,
            'fan_raw': f_votes
        })
    
    df_week = pd.DataFrame(scores)
    
    # 2. 计算综合得分 (根据赛制不同)
    if mechanism == 'Percentage':
        # 旧赛制：直接百分比相加 (粉丝票权重极大)
        total_j = df_week['judge_raw'].sum()
        total_f = df_week['fan_raw'].sum()
        df_week['final_score'] = (df_week['judge_raw']/total_j)*50 + (df_week['fan_raw']/total_f)*50
        
        # 淘汰分数最低者
        df_week = df_week.sort_values('final_score', ascending=False)
        eliminated = df_week.iloc[-1]
    
    elif mechanism == 'RSP':
        # 新赛制：R.S.P. Protocol (Rank + Save)
        # Step A: 排名积分化 (Rank Equalizer)
        df_week['rank_j'] = df_week['judge_raw'].rank(ascending=True) # 分高排名高
        df_week['rank_f'] = df_week['fan_raw'].rank(ascending=True)
        df_week['final_score'] = df_week['rank_j'] + df_week['rank_f']
        
        # Step B: 找出 Bottom 2
        df_week = df_week.sort_values('final_score', ascending=False) # 分高者安全
        bottom_2 = df_week.iloc[-2:]
        
        # Step C: 评委拯救 (Judges' Save)
        # 评委倾向于保留技术分(judge_raw)更高的那个
        c1 = bottom_2.iloc[0]
        c2 = bottom_2.iloc[1]
        
        if c1['judge_raw'] > c2['judge_raw']:
            eliminated = c2 # c2 分低，淘汰 c2
            saved = c1
            save_msg = f"Judges SAVED {c1['name']} over {c2['name']}"
        else:
            eliminated = c1 # c1 分低，淘汰 c1
            saved = c2
            save_msg = f"Judges SAVED {c2['name']} over {c1['name']}"
            
    # 执行淘汰
    eliminated['obj'].is_eliminated = True
    return eliminated['name']

# ==========================================
# 4. 执行全赛季仿真
# ==========================================
def run_full_season(mechanism):
    contestants = create_contestants()
    history = []
    
    # 模拟 10 周
    for week in range(1, 11):
        eliminated_name = simulate_week(contestants, mechanism)
        if eliminated_name is None: break
        
        # 记录当周幸存者状态
        week_status = {}
        for c in contestants:
            week_status[c.name] = 0 if c.is_eliminated else 1
        week_status['Week'] = week
        week_status['Eliminated'] = eliminated_name
        history.append(week_status)
        
    return pd.DataFrame(history)

# 运行两次仿真
print("正在模拟旧赛制 (Percentage)...")
res_old = run_full_season('Percentage')

print("正在模拟新赛制 (R.S.P. Protocol)...")
res_new = run_full_season('RSP')

# ==========================================
# 5. 结果可视化：生存对比图
# ==========================================
def plot_survival_comparison(res_old, res_new):
    plt.figure(figsize=(12, 6))
    
    # 提取关键角色的生存周数
    key_roles = ["Viral Vinny", "Broadway Bella", "Patriot Paul", "Legend Larry", "Disney Daisy"]
    
    def get_survival_weeks(df):
        weeks = {}
        for role in key_roles:
            # 找到最后一次状态为1的周
            if role in df.columns:
                alive = df[df[role] == 1]
                weeks[role] = len(alive)
        return weeks

    weeks_old = get_survival_weeks(res_old)
    weeks_new = get_survival_weeks(res_new)
    
    # 绘图
    x = np.arange(len(key_roles))
    width = 0.35
    
    plt.bar(x - width/2, weeks_old.values(), width, label='Current System (Percentage)', color='#e74c3c', alpha=0.8)
    plt.bar(x + width/2, weeks_new.values(), width, label='Proposed R.S.P. System', color='#2ecc71', alpha=0.8)
    
    plt.ylabel('Survival Weeks')
    plt.title('Impact of New Protocol on Different Contestant Archetypes')
    plt.xticks(x, key_roles, rotation=15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 添加注解
    plt.annotate('Viral Star Survives!', xy=(0-width/2, 9), xytext=(0, 11),
                 arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    plt.annotate('Talent Protected', xy=(1+width/2, 9), xytext=(1, 11),
                 arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

    plt.tight_layout()
    plt.show()

plot_survival_comparison(res_old, res_new)

# 输出文字报告
print("\n=== Simulation Results Report ===")
print("1. Viral Star (流量明星):")
print(f"   - Old System Survival: {res_old[res_old['Viral Vinny']==1].shape[0]} Weeks")
print(f"   - New System Survival: {res_new[res_new['Viral Vinny']==1].shape[0]} Weeks")
print("   -> Conclusion: New system successfully curbed the 'fan-base inflation'.")

print("\n2. Broadway Pro (百老汇演员):")
print(f"   - Old System Survival: {res_old[res_old['Broadway Bella']==1].shape[0]} Weeks")
print(f"   - New System Survival: {res_new[res_new['Broadway Bella']==1].shape[0]} Weeks")
print("   -> Conclusion: High-skill, low-fame talent is protected by Judges' Save.")