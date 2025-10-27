import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import os

from imblearn.over_sampling import SMOTE  # 添加SMOTE导入
import warnings
warnings.filterwarnings("ignore")

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
        
        return precision, sensitivity, specificity
    else:
        # 多分类情况使用sklearn的方法
        precision = precision_score(y_true, y_pred, average='weighted')
        sensitivity = recall_score(y_true, y_pred, average='weighted')
        # 特异性计算较为复杂，在多分类情况下通常不单独报告
        specificity = 0
        
        return precision, sensitivity, specificity

def train_decision_tree_with_pca():
    """
    使用PCA降维和决策树训练模型
    """
    # 加载数据
    X, y = load_and_preprocess_data()
    
    print(f"原始数据形状: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

    
    # 使用PCA进行降维
    # 保留95%的方差信息
    pca = PCA(n_components=0.70)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # # 应用SMOTE过采样到训练数据
    # smote = SMOTE(random_state=42)
    # X_train_pca, y_train = smote.fit_resample(X_train_pca, y_train)
    # print(f"SMOTE后训练集形状: X_train={X_train_pca.shape}, y_train={y_train.shape}")
    # print(f"SMOTE后训练集标签分布: {np.bincount(y_train)}")
    
    print(f"PCA降维后:")
    print(f"训练集形状: X_train_pca={X_train_pca.shape}")
    print(f"测试集形状: X_test_pca={X_test_pca.shape}")
    print(f"保留的主成分数量: {pca.n_components_}")
    print(f"保留的方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 训练决策树模型
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,  # 限制树的深度防止过拟合
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    dt_model.fit(X_train_pca, y_train)
    
    # 预测
    y_train_pred = dt_model.predict(X_train_pca)
    y_test_pred = dt_model.predict(X_test_pca)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== PCA + 决策树模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return dt_model, pca, scaler

def train_decision_tree_with_feature_selection():
    """
    使用特征选择和决策树训练模型
    """
    # 加载数据
    X, y = load_and_preprocess_data()
    
    print(f"\n原始数据形状: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 使用SelectKBest进行特征选择 (选择50个最佳特征)
    selector = SelectKBest(score_func=f_classif, k=20)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"特征选择后:")
    print(f"训练集形状: X_train_selected={X_train_selected.shape}")
    print(f"测试集形状: X_test_selected={X_test_selected.shape}")
    
    # 训练决策树模型
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,  # 限制树的深度防止过拟合
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    dt_model.fit(X_train_selected, y_train)
    
    # 预测
    y_train_pred = dt_model.predict(X_train_selected)
    y_test_pred = dt_model.predict(X_test_selected)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== 特征选择 + 决策树模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return dt_model, selector, scaler

def train_decision_tree_without_reduction():
    """
    不进行降维，直接使用决策树训练模型
    """
    # 加载数据
    X, y = load_and_preprocess_data()
    
    print(f"\n原始数据形状: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练决策树模型
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,  # 限制树的深度防止过拟合
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    dt_model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== 直接决策树模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return dt_model, scaler

if __name__ == '__main__':
    print("=== 决策树模型训练比较 ===")
    
    # 1. 不进行降维的决策树
    model1, scaler1 = train_decision_tree_without_reduction()
    
    # 2. 使用PCA降维的决策树
    model2, pca, scaler2 = train_decision_tree_with_pca()
    
    # 3. 使用特征选择的决策树
    model3, selector, scaler3 = train_decision_tree_with_feature_selection()
    
    print("\n=== 总结 ===")
    print("1. 直接使用决策树: 使用全部133个特征")
    print("2. PCA降维+决策树: 降维到保留95%方差的主成分数量")
    print("3. 特征选择+决策树: 从133个特征中选择50个最佳特征")
    print("\n对于133维特征和155个样本的数据集:")
    print("- 降维有助于减少过拟合风险")
    print("- PCA适用于发现数据中的主要变化方向")
    print("- 特征选择适用于保留原始特征的可解释性")