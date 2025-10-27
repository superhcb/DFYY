from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import os
import numpy as np
import pandas as pd


def load_and_preprocess_data():
    """
    加载并预处理Excel数据
    C列到EE列作为输入特征 (共133列)
    EF列作为输出标签
    """
    # 读取Excel文件
    file_path = os.path.join('data', 'classified2_data.xlsx')
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
    
    # 提取C列到EE列作为输入特征 (索引2到134)
    X = df.iloc[:, 2:135].values  # C列到EE列，共133列
    
    # 提取EF列作为标签 (索引135)
    y = df.iloc[:, 135].values    # EF列
    
    # 处理缺失值
    # 将NaN替换为0
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    
    # 将标签转换为整数类型，并确保从0开始编号
    y = y.astype(int)
    # 如果标签是1-4级，转换为0-3级
    if np.min(y) == 1:
        y = y - 1
    
    return X, y

def calculate_metrics(y_true, y_pred):
    """
    计算精确率、特异性、敏感性
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 对于二分类问题
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        # 精确率 (Precision) = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # 敏感性 (Sensitivity/Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 特异性 (Specificity) = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 约登指数 (Youden's Index) = Sensitivity + Specificity - 1
        youden_index = sensitivity + specificity - 1
        
        return precision, sensitivity, specificity, youden_index
    else:
        # 多分类情况使用sklearn的方法
        precision = precision_score(y_true, y_pred, average='weighted')
        sensitivity = recall_score(y_true, y_pred, average='weighted')
        # 特异性计算较为复杂，在多分类情况下通常不单独报告
        specificity = 0
        # 约登指数主要用于二分类问题
        youden_index = 0
        
        return precision, sensitivity, specificity, youden_index



