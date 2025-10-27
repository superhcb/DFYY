import pandas as pd
import os

# 读取Excel文件
file_path = os.path.join('data', 'dwx_preprocessed.xlsx')
df = pd.read_excel(file_path, sheet_name='预处理后数据')

# 根据"冠状动脉造影结果（狭窄程度%）"列创建新的分级列
def classify_stenosis_degree(stenosis_percentage):
    """
    根据狭窄百分比对冠状动脉狭窄程度进行分级：
    0-49: 1级
    50-74: 2级
    75-99: 3级
    100: 4级
    """
    # if pd.isna(stenosis_percentage):
    #     return None
    # elif 0 <= stenosis_percentage <= 49:
    #     return 1
    # elif 50 <= stenosis_percentage <= 74:
    #     return 2
    # elif 75 <= stenosis_percentage <= 99:
    #     return 3
    # elif stenosis_percentage == 100:
    #     return 4
    # else:
    #     return None
    if pd.isna(stenosis_percentage):
        return None
    elif 0 <= stenosis_percentage <= 74:
        return 1
    elif 75 <= stenosis_percentage <= 100:
        return 2
    else:
        return None

# 应用分级函数
df['冠状动脉狭窄程度分级'] = df['冠状动脉造影结果（狭窄程度%）'].apply(classify_stenosis_degree)

# 统计各级别数量
classification_counts = df['冠状动脉狭窄程度分级'].value_counts().sort_index()

# 创建新的DataFrame用于输出
output_df = df.copy()

# 将结果保存到新的Excel文件
output_file_path = os.path.join('data', 'classified2_data.xlsx')
output_df.to_excel(output_file_path, index=False)

# print(f"已成功创建新的Excel文件: {output_file_path}")
# print("\n冠状动脉狭窄程度分级统计:")
# print("1级(0-49%):  {}个".format(classification_counts.get(1, 0)))
# print("2级(50-74%): {}个".format(classification_counts.get(2, 0)))
# print("3级(75-99%): {}个".format(classification_counts.get(3, 0)))
# print("4级(100%):   {}个".format(classification_counts.get(4, 0)))
# print("\n总计: {}个".format(classification_counts.sum()))

print(f"已成功创建新的Excel文件: {output_file_path}")
print("\n冠状动脉狭窄程度分级统计:")
print("1级(0-74%):  {}个".format(classification_counts.get(1, 0)))
print("2级(75-100%): {}个".format(classification_counts.get(2, 0)))
print("\n总计: {}个".format(classification_counts.sum()))