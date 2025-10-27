import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
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

def train_svm_with_pca():
    """
    使用PCA降维和支持向量机训练模型
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
    
    # 数据标准化（SVM和PCA都需要标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # 注意：只变换，不重新拟合
    
    # 使用PCA进行降维（保留90%的方差）
    pca = PCA(n_components=0.7)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)  # 注意：只变换，不重新拟合
    
    print(f"PCA降维后:")
    print(f"训练集形状: X_train_pca={X_train_pca.shape}")
    print(f"测试集形状: X_test_pca={X_test_pca.shape}")
    print(f"保留的主成分数量: {pca.n_components_}")
    print(f"保留的方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 训练支持向量机模型
    svm_model = SVC(
        kernel='rbf',          # 使用RBF核函数
        C=1.0,                 # 正则化参数
        gamma='scale',         # 核函数系数
        random_state=42,       # 随机种子
        probability=True       # 启用概率预测
    )
    
    # 在PCA降维后的数据上训练模型
    svm_model.fit(X_train_pca, y_train)
    
    # 预测
    y_train_pred = svm_model.predict(X_train_pca)
    y_test_pred = svm_model.predict(X_test_pca)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== PCA + 支持向量机模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return svm_model, pca, scaler

def train_svm_without_pca():
    """
    不进行降维，直接使用支持向量机训练模型
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
    
    # 训练支持向量机模型
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    )
    
    svm_model.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = svm_model.predict(X_train_scaled)
    y_test_pred = svm_model.predict(X_test_scaled)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== 直接支持向量机模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return svm_model, scaler

def train_svm_with_pca_linear():
    """
    使用PCA降维和线性核支持向量机训练模型
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
    
    # 使用PCA进行降维（保留95%的方差）
    pca = PCA(n_components=0.7)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA降维后(保留95%方差):")
    print(f"训练集形状: X_train_pca={X_train_pca.shape}")
    print(f"测试集形状: X_test_pca={X_test_pca.shape}")
    print(f"保留的主成分数量: {pca.n_components_}")
    print(f"保留的方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 训练线性核支持向量机模型
    svm_model = SVC(
        kernel='linear',       # 使用线性核函数
        C=1.0,
        random_state=42,
        probability=True
    )
    
    svm_model.fit(X_train_pca, y_train)
    
    # 预测
    y_train_pred = svm_model.predict(X_train_pca)
    y_test_pred = svm_model.predict(X_test_pca)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== PCA + 线性核支持向量机模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return svm_model, pca, scaler

if __name__ == '__main__':
    print("=== 支持向量机模型训练比较 ===")
    
    # 1. 不进行降维的支持向量机
    model1, scaler1 = train_svm_without_pca()
    
    # 2. 使用PCA降维的RBF核支持向量机
    model2, pca, scaler2 = train_svm_with_pca()
    
    # 3. 使用PCA降维的线性核支持向量机
    model3, pca3, scaler3 = train_svm_with_pca_linear()
    
    print("\n=== 总结 ===")
    print("1. 直接支持向量机: 使用全部133个原始特征")
    print("2. PCA降维+RBF核SVM: 降维到保留90%方差的主成分")
    print("3. PCA降维+线性核SVM: 降维到保留95%方差的主成分")
    print("\n对于133维特征和155个样本的数据集:")
    print("- PCA降维可以减少特征维度，防止过拟合")
    print("- SVM对高维数据处理能力强，但降维可以提高训练速度")
    print("- 线性核适合线性可分问题，RBF核适合非线性问题")
    print("- 三种方法各有优势，可以通过交叉验证选择更好的模型")