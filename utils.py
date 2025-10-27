from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_curve, auc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    feature_names = df.columns[2:135].tolist()  # 获取特征名称
    
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
    
    return X, y, feature_names

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
    
def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """
    绘制ROC曲线
    
    参数:
    y_true: 真实标签
    y_scores: 预测概率分数
    title: 图形标题
    """
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    绘制混淆矩阵
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    title: 图形标题
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1'], 
                yticklabels=['0', '1'])
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return cm

def plot_feature_importance(feature_importances, feature_names=None, top_n=20, title="Feature Importance"):
    """
    绘制特征重要性图表
    
    参数:
    feature_importances: 特征重要性数组
    feature_names: 特征名称列表（可选）
    top_n: 显示前N个最重要的特征
    title: 图表标题
    """
    # 获取特征重要性排序的索引
    indices = np.argsort(feature_importances)[::-1]
    
    # 只显示前top_n个特征
    top_indices = indices[:top_n]
    top_importances = feature_importances[top_indices]
    
    # 设置特征名称
    if feature_names is not None:
        top_feature_names = [feature_names[i] for i in top_indices]
    else:
        top_feature_names = [f"Feature {i}" for i in top_indices]
    
    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(range(top_n), top_importances, align="center")
    plt.xticks(range(top_n), top_feature_names, rotation=45, ha="right")
    plt.xlim([-1, top_n])
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

