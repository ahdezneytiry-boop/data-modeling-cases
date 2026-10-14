import pandas as pd
import os

# 配置你的文件路径（和主代码一致）
FILES = {
    "核心数据(processed_dancing_data_final.csv)": "E:\\MCM\\daima\\processed_dancing_data_final.csv",
    "特征数据(features.csv)": "E:\\MCM\\daima\\features.csv",
    "标签数据(labels.csv)": "E:\\MCM\\daima\\labels.csv"
}

# 查看每个文件的列名
for file_desc, file_path in FILES.items():
    print(f"="*50)
    print(f"📋 {file_desc} 的所有列名：")
    print(f"文件路径：{file_path}")
    try:
        # 用gbk编码读取（若报错，换成encoding='utf-8'）
        df = pd.read_csv(file_path, encoding="gbk", nrows=1)  # 只读1行，快速获取列名
        columns = df.columns.tolist()
        print(f"共{len(columns)}个列：")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")
        
        # 重点提示：寻找关键字段（选手名、赛季、排名、票数）
        key_field_types = {
            "选手名称字段": ["name", "contestant", "celebrity", "id"],  # 可能包含这些关键词
            "赛季字段": ["season", "session", "year"],
            "排名字段": ["rank", "placement", "result"],
            "粉丝票数字段": ["vote", "fan", "support"]
        }
        print(f"\n🔍 重点字段识别（请核对）：")
        for field_type, keywords in key_field_types.items():
            matched_cols = [col for col in columns if any(kw in col.lower() for kw in keywords)]
            if matched_cols:
                print(f"  {field_type}：{matched_cols}")
            else:
                print(f"  {field_type}：未找到（请手动确认）")
        
    except Exception as e:
        print(f"❌ 读取失败：{str(e)}")
        print(f"  建议尝试修改编码：将 encoding='gbk' 改为 encoding='utf-8'")
    print()