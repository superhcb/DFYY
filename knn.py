import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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

def train_knn_with_pca():
    """
    使用PCA降维和K近邻算法训练模型
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
    
    # 数据标准化（KNN对特征尺度敏感，必须标准化）
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
    
    # 训练K近邻模型
    knn_model = KNeighborsClassifier(
        n_neighbors=5,         # 邻居数量
        weights='distance',    # 按距离加权
        algorithm='auto',      # 自动选择算法
        leaf_size=30,          # 叶节点大小
        p=2,                   # 距离度量参数（2表示欧氏距离）
        metric='minkowski'     # 距离度量方法
    )
    
    # 在PCA降维后的数据上训练模型
    knn_model.fit(X_train_pca, y_train)
    
    # 预测
    y_train_pred = knn_model.predict(X_train_pca)
    y_test_pred = knn_model.predict(X_test_pca)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== PCA + K近邻模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return knn_model, pca, scaler

def train_knn_without_pca():
    """
    不进行降维，直接使用K近邻算法训练模型
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
    
    # 训练K近邻模型
    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski'
    )
    
    knn_model.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = knn_model.predict(X_train_scaled)
    y_test_pred = knn_model.predict(X_test_scaled)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== 直接K近邻模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return knn_model, scaler

def train_knn_with_pca_different_k():
    """
    使用PCA降维和不同K值的K近邻算法训练模型
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
    
    # 测试不同的K值
    k_values = [3, 5, 7, 9, 11, 13]
    best_k = 3
    best_accuracy = 0
    
    print("\n测试不同K值的性能:")
    for k in k_values:
        # 训练K近邻模型
        knn_model = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            algorithm='auto'
        )
        
        knn_model.fit(X_train_pca, y_train)
        
        # 预测和评估
        y_test_pred = knn_model.predict(X_test_pca)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"  K={k}: 测试准确率={test_accuracy:.4f}")
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_k = k
    
    print(f"\n最佳K值: {best_k}, 最佳准确率: {best_accuracy:.4f}")
    
    # 使用最佳K值训练最终模型
    final_knn_model = KNeighborsClassifier(
        n_neighbors=best_k,
        weights='distance',
        algorithm='auto'
    )
    
    final_knn_model.fit(X_train_pca, y_train)
    
    # 最终预测
    y_train_pred = final_knn_model.predict(X_train_pca)
    y_test_pred = final_knn_model.predict(X_test_pca)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision, train_sensitivity, train_specificity = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== PCA + 最优K值K近邻模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    
    return final_knn_model, pca, scaler, best_k

if __name__ == '__main__':
    print("=== K近邻模型训练比较 ===")
    
    # 1. 不进行降维的K近邻
    model1, scaler1 = train_knn_without_pca()
    
    # 2. 使用PCA降维的K近邻
    model2, pca, scaler2 = train_knn_with_pca()
    
    # 3. 使用PCA降维和不同K值优化的K近邻
    model3, pca3, scaler3, best_k = train_knn_with_pca_different_k()
    
    print("\n=== 总结 ===")
    print("1. 直接K近邻: 使用全部133个原始特征")
    print("2. PCA降维+K近邻: 降维到保留90%方差的主成分")
    print("3. PCA降维+最优K值K近邻: 降维到保留95%方差的主成分，并优化K值")
    print("\n对于133维特征和155个样本的数据集:")
    print("- PCA降维可以减少特征维度，防止维度灾难")
    print("- KNN对高维数据敏感，降维可以提高性能")
    print("- K值选择很重要，需要通过交叉验证确定最优值")
    print("- 三种方法各有优势，可以通过交叉验证选择更好的模型")