from utils import calculate_metrics, load_and_preprocess_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
        X_scaled, y, test_size=0.3, random_state=808, stratify=y
    )

    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

    
    # 使用PCA进行降维
    # 保留95%的方差信息
    pca = PCA(n_components=0.6)
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
        max_depth=7,  # 限制树的深度防止过拟合
        min_samples_split=8,
        min_samples_leaf=1,
        random_state=808
    )
    
    dt_model.fit(X_train_pca, y_train)
    
    # 预测
    y_train_pred = dt_model.predict(X_train_pca)
    y_test_pred = dt_model.predict(X_test_pca)
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision, train_sensitivity, train_specificity, train_ydindex = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity, test_ydindex = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== PCA + 决策树模型结果 ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    print(f"测试集约登指数: {test_ydindex:.4f}")
    
    return dt_model, pca, scaler


if __name__ == "__main__":
    train_decision_tree_with_pca()

