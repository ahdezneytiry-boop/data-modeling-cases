import pandas as pd
import os

# 步骤1：导入数据（使用原始字符串处理路径，避免转义问题）
# 建议将文件路径修改为实际存放位置
spd_path = r"E:\华数杯\2025年第六届华数杯数学建模竞赛赛题\C题\附录.csv"
cie_path = r"E:\华数杯\2025年第六届华数杯数学建模竞赛赛题\C题\CIE_xyz_1931_2deg.csv"

# 检查文件是否存在
if not os.path.exists(spd_path):
    raise FileNotFoundError(f"SPD数据文件不存在：{spd_path}")
if not os.path.exists(cie_path):
    raise FileNotFoundError(f"CIE数据文件不存在：{cie_path}")

# 读取数据时指定列名，确保列名统一
spd_data = pd.read_csv(
    spd_path,
    names=["波长", "光谱功率(W/nm)"],  # 强制指定列名，避免原始文件列名问题
    skiprows=1  # 跳过原始文件的表头行
)

cie_data = pd.read_csv(
    cie_path,
    names=["波长", "x_bar", "y_bar", "z_bar"],  # 强制指定列名
    skiprows=1  # 跳过原始文件的表头行
)

# 步骤2：波长范围对齐（截取CIE数据中380-780nm的部分）
cie_filtered = cie_data[(cie_data["波长"] >= 380) & (cie_data["波长"] <= 780)]

# 步骤3：合并数据集（以波长为匹配键）
merged_data = pd.merge(
    spd_data,
    cie_filtered,
    on="波长",
    how="inner"
)

# 步骤4：保存合并后的新表格
output_path = "SPD_CIE.csv"
merged_data.to_csv(output_path, index=False)

# 打印合并结果信息
if not merged_data.empty:
    print(f"合并后的数据包含{len(merged_data)}条记录，波长范围：{merged_data['波长'].min()}-{merged_data['波长'].max()}nm")
    print("合并后的数据列名：", merged_data.columns.tolist())
else:
    print("警告：合并后的数据为空，请检查两个文件的波长范围是否有重叠")