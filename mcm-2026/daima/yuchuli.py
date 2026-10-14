import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler  # 替换StandardScaler为MinMaxScaler
from sklearn.compose import ColumnTransformer

# -------------------------- 基础配置（核心修改：修正标识列+新增特征/标签保存路径） --------------------------
INPUT_PATH = r"E:\MCM\daima\2026_MCM_Problem_C_Data.csv"
OUTPUT_PATH = r"E:\MCM\daima\processed_dancing_data_final.csv"
FEATURES_PATH = r"E:\MCM\daima\features.csv"  # 单独保存建模特征
LABELS_PATH = r"E:\MCM\daima\labels.csv"      # 单独保存标签
SCORE_COL_KEYS = ['week', 'score']  # 分数列筛选关键词
# 修正标识列：移除labels列results，仅保留选手/舞伴纯标识
ID_COLS = ['celebrity_name', 'ballroom_partner']  
LABEL_COL = 'results'  # 单独定义标签列，便于分离/保存
# 固定分类列（业务特征）
CAT_COLS = ['celebrity_industry', 'celebrity_homecountry_region', 'celebrity_homestate']

# -------------------------- 工具函数（保留原功能，无修改） --------------------------
def clean_col_name(col_name):
    """清洗列名：特殊字符转下划线+统一小写+去连续下划线/首尾下划线"""
    if pd.isna(col_name):
        return 'unknown_col'
    col_name = str(col_name).replace('/', '_').replace(' ', '_').replace('-', '_')
    col_name = re.sub(r'__+', '_', col_name)
    col_name = col_name.strip('_').lower()
    return col_name

def normalize_category_val(val):
    """规范化分类值：去首尾空格+首字母大写，避免大小写冗余"""
    if pd.isna(val) or val in ['Unknown', 'N/A']:
        return 'Unknown'
    val = str(val).strip()
    val = ' '.join([word.capitalize() for word in val.split()])
    return val

# -------------------------- 1. 数据加载（增强鲁棒性，无修改） --------------------------
try:
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
    print("="*60)
    print(f"数据加载成功！原始数据形状：{df.shape}（行：{df.shape[0]}, 列：{df.shape[1]}）")
    # 源头清洗列名，解决"名称怪怪的"问题
    df.columns = [clean_col_name(col) for col in df.columns]
    ID_COLS = [clean_col_name(col) for col in ID_COLS]
    LABEL_COL = clean_col_name(LABEL_COL)
    CAT_COLS = [clean_col_name(col) for col in CAT_COLS]
    print(f"原始列名已规范化，前5列：{list(df.columns)[:5]}")
except FileNotFoundError:
    print(f"错误：未找到文件！{INPUT_PATH}")
    exit()
except Exception as e:
    print(f"数据加载失败：{str(e)}")
    exit()

# -------------------------- 2. 自动删除重复行（新增核心功能） --------------------------
print("="*60)
print("开始重复行处理：自动删除完全重复的行...")
dup_rows = df.duplicated().sum()
if dup_rows > 0:
    df = df.drop_duplicates(reset_index=True, drop=True)  # 删除后重置索引
    print(f"✅ 删除{dup_rows}行完全重复数据，剩余数据形状：{df.shape}")
else:
    print(f"✅ 无重复数据，数据唯一性验证通过")

# -------------------------- 3. 数据类型优化（核心修改：自动识别所有数值列） --------------------------
print("="*60)
print("开始数据类型优化+自动识别数值列...")
# 全局筛选分数列（业务特征，后续单独处理缺失值）
score_cols = [col for col in df.columns if all(key in col for key in SCORE_COL_KEYS)]
# 步骤1：分类列转category类型（固定）
for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].apply(normalize_category_val).astype('category')
# 步骤2：标签列转category（建模目标）
if LABEL_COL in df.columns:
    df[LABEL_COL] = df[LABEL_COL].astype('category')
# 步骤3：自动识别**所有数值列**（排除分类列/标识列/标签列），避免遗漏
all_num_cols = [
    col for col in df.columns
    if pd.api.types.is_numeric_dtype(df[col])  # 自动识别数值类型（int/float）
    and col not in CAT_COLS + ID_COLS + [LABEL_COL]  # 排除非数值列
]
# 步骤4：所有数值列强制转浮点型，非数值转NaN（统一缺失值）
for col in all_num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# 拆分数值列：分数列（score_cols）+ 其他数值列（other_num_cols），便于差异化缺失值处理
other_num_cols = [col for col in all_num_cols if col not in score_cols]

# 打印类型优化结果
print(f"✅ 分类列（{len(CAT_COLS)}个）：{[col for col in CAT_COLS if col in df.columns]}")
print(f"✅ 分数数值列（{len(score_cols)}个）：前3列{score_cols[:3]}..." if score_cols else "✅ 无分数列")
print(f"✅ 其他数值列（{len(other_num_cols)}个）：{other_num_cols}" if other_num_cols else "✅ 无其他数值列")
print(f"✅ 所有数值列（{len(all_num_cols)}个）：已统一转浮点型，量纲一致处理")

# -------------------------- 4. 缺失值处理（核心修改：精细化差异化填充） --------------------------
print("="*60)
print("开始缺失值精细化处理（贴合业务逻辑，无信息损失）...")
# 统一将N/A/na等转为np.nan，便于后续处理
df = df.replace(['N/A', 'na', 'NA', 'null'], np.nan, regex=False)

# 4.1 分类列缺失值：固定填充Unknown
for col in CAT_COLS:
    if col not in df.columns:
        continue
    missing_num = df[col].isna().sum()
    if missing_num > 0:
        df[col] = df[col].fillna('Unknown').astype('category')
        print(f"分类列{col}：填充{missing_num}个缺失值 → Unknown")
    else:
        print(f"分类列{col}：无缺失值")

# 4.2 分数列缺失值：填充0（业务逻辑：未参赛/淘汰）
print("-"*50)
for col in score_cols:
    if col not in df.columns:
        continue
    missing_num = df[col].isna().sum()
    if missing_num > 0:
        df[col] = df[col].fillna(0)
        print(f"分数列{col}：填充{missing_num}个缺失值 → 0（代表未参赛/淘汰）")
    else:
        print(f"分数列{col}：无缺失值")

# 4.3 其他数值列缺失值：填充**列中位数**（更严谨，避免均值受异常值影响）
for col in other_num_cols:
    if col not in df.columns:
        continue
    missing_num = df[col].isna().sum()
    if missing_num > 0:
        col_median = df[col].median()  # 按列计算中位数，单独填充
        df[col] = df[col].fillna(col_median)
        print(f"其他数值列{col}：填充{missing_num}个缺失值 → 列中位数{col_median:.2f}")
    else:
        print(f"其他数值列{col}：无缺失值")

# 全局缺失值统计
total_missing = df.isna().sum().sum()
missing_ratio = total_missing / (df.shape[0] * df.shape[1]) * 100
print(f"✅ 缺失值处理完成！全局剩余缺失值：{total_missing}个，占比：{missing_ratio:.6f}%")

# -------------------------- 5. 异常值处理（优化：覆盖所有数值列，保留0值业务逻辑） --------------------------
print("="*60)
print("开始异常值处理（IQR方法）→ 分数列仅检测参赛选手，其他数值列检测全部有效数据...")
# 处理分数列：仅筛选>0的参赛数据，0值（淘汰）不参与异常值检测
for col in score_cols:
    if col not in df.columns:
        continue
    valid_data = df[col][df[col] > 0].dropna()
    if len(valid_data) < 10:
        print(f"列{col}：有效数据仅{len(valid_data)}条，跳过异常值检测")
        continue
    Q1, Q3 = valid_data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] > 0) & ((df[col] < lower_bound) | (df[col] > upper_bound))]
    outlier_num = len(outliers)
    if outlier_num > 0:
        valid_median = valid_data.median()
        df.loc[outliers.index, col] = valid_median
        outlier_ratio = outlier_num / len(valid_data) * 100
        print(f"分数列{col}：参赛{len(valid_data)}人｜异常值{outlier_num}个（{outlier_ratio:.2f}%）｜替换为参赛中位数{valid_median:.2f}")
    else:
        print(f"分数列{col}：参赛{len(valid_data)}人｜无异常值")

# 处理其他数值列：检测所有非缺失有效数据，无0值业务限制
for col in other_num_cols:
    if col not in df.columns:
        continue
    valid_data = df[col].dropna()
    if len(valid_data) < 10:
        print(f"列{col}：有效数据仅{len(valid_data)}条，跳过异常值检测")
        continue
    Q1, Q3 = valid_data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_num = len(outliers)
    if outlier_num > 0:
        col_median = valid_data.median()
        df.loc[outliers.index, col] = col_median
        outlier_ratio = outlier_num / len(valid_data) * 100
        print(f"其他数值列{col}：有效{len(valid_data)}条｜异常值{outlier_num}个（{outlier_ratio:.2f}%）｜替换为列中位数{col_median:.2f}")
    else:
        print(f"其他数值列{col}：有效{len(valid_data)}条｜无异常值")
print(f"✅ 异常值处理完成！所有数值列异常值已修正，业务逻辑保留")

# -------------------------- 6. 特征分离（核心修改：严格区分标识/特征/标签） --------------------------
print("="*60)
print("开始特征分离（标识列/建模特征/标签列 严格区分）...")
# 校验标签列是否存在
if LABEL_COL not in df.columns:
    print(f"❌ 错误：未找到标签列{LABEL_COL}，无法分离特征和标签！")
    exit()
# 建模特征X：排除【标识列+标签列】，仅保留分类列+所有数值列
feature_cols = [col for col in df.columns if col not in ID_COLS + [LABEL_COL]]
X = df[feature_cols].copy()
# 标签列y：单独提取，后续单独保存
y = df[LABEL_COL].copy()
# 标识列：单独提取，可随特征一起保存用于溯源
id_df = df[ID_COLS].copy()

print(f"✅ 标识列（{len(ID_COLS)}个）：{ID_COLS}（用于结果溯源，不参与建模）")
print(f"✅ 建模特征集X形状：{X.shape}（行：{X.shape[0]}, 特征列：{X.shape[1]}）")
print(f"✅ 标签列y形状：{y.shape}（{len(y.unique())}个类别）")

# -------------------------- 7. 特征转换（核心修改：MinMaxScaler缩放到0-10区间） --------------------------
print("="*60)
print("开始特征转换（分类列OneHot+所有数值列0-10归一化）...")
# 筛选特征集中的分类列和数值列（避免列名缺失）
X_cat_cols = [col for col in CAT_COLS if col in X.columns]
X_num_cols = [col for col in all_num_cols if col in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        # OneHot编码：drop='first'避免多重共线性，handle_unknown忽略测试集新类别
        ('cat', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), X_cat_cols),
        # 核心修改：MinMaxScaler缩放到0-10区间（贴合裁判打分业务，解决负数问题）
        ('num', MinMaxScaler(feature_range=(0, 10)), X_num_cols)
    ],
    remainder='passthrough',  # 保留其他可能的特征列，避免遗漏
    verbose_feature_names_out=True
)
# 执行转换，得到建模可用的特征矩阵
processed_X = preprocessor.fit_transform(X)
print(f"✅ 特征转换完成！处理后特征形状：{processed_X.shape}（行：{processed_X.shape[0]}, 总特征数：{processed_X.shape[1]}）")

# -------------------------- 8. 特征名称最终规范化（保留原功能，无修改） --------------------------
print("="*60)
print("开始特征名称最终规范化（小写+下划线+无特殊字符）...")
processed_col_names = preprocessor.get_feature_names_out()
processed_col_names = [clean_col_name(name) for name in processed_col_names]
# 防重复列名，添加索引区分
processed_col_names = [f"{name}_{i}" if processed_col_names.count(name) > 1 else name 
                       for i, name in enumerate(processed_col_names)]
print(f"✅ 特征名规范化完成，前5个特征：{processed_col_names[:5]}")

# -------------------------- 9. 构建特征/标签数据框（用于单独保存） --------------------------
print("="*60)
print("构建特征/标签数据框（含标识列溯源，单独保存）...")
# 处理后特征数据框：加入标识列，方便后续结果分析/溯源
processed_features = pd.DataFrame(processed_X, columns=processed_col_names, index=df.index)
processed_features = pd.concat([id_df, processed_features], axis=1)  # 标识列在前，便于查看
# 标签数据框：加入标识列，保证特征和标签一一对应（可根据需要选择是否保留）
processed_labels = pd.concat([id_df, y], axis=1)
# 重置索引，避免混乱
processed_features = processed_features.reset_index(drop=True)
processed_labels = processed_labels.reset_index(drop=True)

# -------------------------- 10. 单独保存features.csv和labels.csv（核心修改） --------------------------
print("="*60)
print("开始单独保存特征和标签文件（防混存，建模直接调用）...")
try:
    # 保存建模特征：features.csv（含标识列，无索引，防乱码）
    processed_features.to_csv(FEATURES_PATH, index=False, encoding='utf-8-sig')
    # 保存标签：labels.csv（含标识列，一一对应，无索引，防乱码）
    processed_labels.to_csv(LABELS_PATH, index=False, encoding='utf-8-sig')
    # 可选：保存全量处理数据（保留原功能）
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print(f"✅ 建模特征保存成功：\n{FEATURES_PATH}（形状：{processed_features.shape}）")
    print(f"✅ 标签保存成功：\n{LABELS_PATH}（形状：{processed_labels.shape}）")
    print(f"✅ 全量处理数据保存成功：\n{OUTPUT_PATH}（形状：{df.shape}）")
except Exception as e:
    print(f"❌ 数据保存失败：{str(e)}")
    print("提示：若报权限错误，关闭所有打开该文件夹的Excel/记事本程序")
    exit()

# -------------------------- 11. 数据完整性全面验证（适配新修改，竞赛汇报专用） --------------------------
print("="*60)
print("【🔍 数据完整性最终验证（MCM竞赛汇报专用）】")
# 1. 特征/标签行数一致性（核心：必须完全一致）
row_consistent = len(processed_features) == len(processed_labels) == len(df)
# 2. 特征集缺失值统计
feat_missing = processed_features.isna().sum().sum()
feat_missing_ratio = feat_missing / (processed_features.shape[0] * processed_features.shape[1]) * 100
# 3. 标签列无缺失（建模要求）
label_missing = processed_labels[LABEL_COL].isna().sum()
# 4. 列类型分布
feat_cat_num = len([col for col in processed_features.columns if pd.api.types.is_categorical_dtype(processed_features[col])])
feat_num_num = len([col for col in processed_features.columns if pd.api.types.is_numeric_dtype(processed_features[col])])
# 5. 标识列完整性
id_complete = all([col in processed_features.columns and col in processed_labels.columns for col in ID_COLS])

# 量化输出验证结果
print(f"1. 行数一致性：{row_consistent}（特征：{len(processed_features)}｜标签：{len(processed_labels)}｜全量：{len(df)}）")
print(f"2. 特征集剩余缺失值：{feat_missing}个（占比：{feat_missing_ratio:.6f}%）")
print(f"3. 标签列缺失值：{label_missing}个（建模要求：必须为0）")
print(f"4. 特征集列类型：分类列{feat_cat_num}个｜数值列{feat_num_num}个｜总列{len(processed_features.columns)}")
print(f"5. 标识列完整性：{id_complete}（特征/标签均包含：{ID_COLS}）")

# -------------------------- 12. 处理后数据预览（格式化输出） --------------------------
print("="*60)
print("【📊 处理后特征前5行预览（数值保留3位小数，仅展示前8个数值特征+标识列）】")
# 筛选数值列+标识列预览，避免展示过长
preview_feat_cols = [col for col in processed_features.columns if pd.api.types.is_numeric_dtype(processed_features[col])][:8] + ID_COLS
print(processed_features[preview_feat_cols].head().round(3))
print("="*60)
print("🎉 数据预处理全流程完成！无报错！")
print("📌 建模直接调用：features.csv（特征） + labels.csv（标签），两者行一一对应，含标识列溯源！")
print("📌 所有数值列已缩放到0-10区间，贴合裁判打分业务场景，无负数问题！")