import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import os
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

def train_rf_with_pca():
    """
    使用PCA降维和随机森林训练模型
    """
    # 加载数据
    X, y = load_and_preprocess_data()
    
    print(f"原始数据形状: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 数据标准化（PCA前需要标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # 注意：只变换，不重新拟合
    
    # 使用PCA进行降维（保留90%的方差）
    pca = PCA(n_components=0.70)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)  # 注意：只变换，不重新拟合
    
    print(f"PCA降维后:")
    print(f"训练集形状: X_train_pca={X_train_pca.shape}")
    print(f"测试集形状: X_test_pca={X_test_pca.shape}")
    print(f"保留的主成分数量: {pca.n_components_}")
    print(f"保留的方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 训练随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=100,      # 决策树的数量
        max_depth=10,          # 树的最大深度
        min_samples_split=5,   # 分割内部节点所需的最小样本数
        min_samples_leaf=2,    # 叶节点所需的最小样本数
        random_state=42,       # 随机种子
        n_jobs=-1              # 使用所有可用的CPU核心
    )
    
    # 在PCA降维后的数据上训练模型
    rf_model.fit(X_train_pca, y_train)
    
    # 预测
    y_train_pred = rf_model.predict(X_train_pca)
    y_test_pred = rf_model.predict(X_test_pca)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== PCA + 随机森林模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    # 输出特征重要性（前10个最重要的主成分）
    feature_importance = rf_model.feature_importances_
    important_features_idx = np.argsort(feature_importance)[::-1][:10]
    print(f"\n前10个最重要主成分的贡献度:")
    for i, idx in enumerate(important_features_idx):
        print(f"  PC{idx+1}: {feature_importance[idx]:.4f}")
    
    return rf_model, pca, scaler

def train_rf_without_pca():
    """
    不进行降维，直接使用随机森林训练模型
    """
    # 加载数据
    X, y = load_and_preprocess_data()
    
    print(f"\n原始数据形状: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = rf_model.predict(X_train_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== 直接随机森林模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    # 输出特征重要性（前10个最重要的原始特征）
    feature_importance = rf_model.feature_importances_
    important_features_idx = np.argsort(feature_importance)[::-1][:10]
    print(f"\n前10个最重要原始特征的索引:")
    for i, idx in enumerate(important_features_idx):
        print(f"  特征{idx}: {feature_importance[idx]:.4f}")
    
    return rf_model, scaler

if __name__ == '__main__':
    print("=== 随机森林模型训练比较 ===")
    
    # 1. 不进行降维的随机森林
    model1, scaler1 = train_rf_without_pca()
    
    # 2. 使用PCA降维的随机森林
    model2, pca, scaler2 = train_rf_with_pca()
    
    print("\n=== 总结 ===")
    print("1. 直接随机森林: 使用全部133个原始特征")
    print("2. PCA降维+随机森林: 降维到保留90%方差的主成分")
    print("\n对于133维特征和155个样本的数据集:")
    print("- PCA降维可以减少特征维度，防止过拟合")
    print("- 随机森林本身具有抗过拟合能力，对高维数据也有较好的处理能力")
    print("- 两种方法各有优势，可以通过交叉验证选择更好的模型")