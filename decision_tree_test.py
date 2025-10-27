from utils import calculate_metrics, load_and_preprocess_data, plot_roc_curve, plot_confusion_matrix, plot_feature_importance
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_decision_tree_without_pca():
    """
    不使用PCA降维，直接训练决策树模型以获得原始特征重要性
    """
    # 加载数据
    X, y, feature_names = load_and_preprocess_data()
    
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
    print(f"训练集标签分布: {np.bincount(y_train)}")
    print(f"测试集标签分布: {np.bincount(y_test)}")
    
    # 训练决策树模型（不使用PCA）
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=7,  # 限制树的深度防止过拟合
        min_samples_split=8,
        min_samples_leaf=1,
        random_state=808
    )
    
    dt_model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)

    # 获取预测概率（用于ROC曲线）
    y_train_proba = dt_model.predict_proba(X_train)[:, 1]
    y_test_proba = dt_model.predict_proba(X_test)[:, 1]
    
    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision, train_sensitivity, train_specificity, train_ydindex = calculate_metrics(y_train, y_train_pred)
    test_precision, test_sensitivity, test_specificity, test_ydindex = calculate_metrics(y_test, y_test_pred)
    
    print("\n=== 决策树模型结果（无PCA） ===")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集精确率: {test_precision:.4f}")
    print(f"测试集敏感性: {test_sensitivity:.4f}")
    print(f"测试集特异性: {test_specificity:.4f}")
    print(f"测试集约登指数: {test_ydindex:.4f}")

    # 绘制ROC曲线
    auc_score = plot_roc_curve(y_test, y_test_proba, title="Decision Tree ROC Curve (No PCA)")
    print(f"测试集AUC值: {auc_score:.4f}")

    # 显示混淆矩阵
    plot_confusion_matrix(y_test, y_test_pred, title="Decision Tree Confusion Matrix (No PCA)")

    # 显示原始特征重要性
    plot_feature_importance(dt_model.feature_importances_, feature_names, top_n=20, title="Top 20 Original Features Importance")
    print("\n原始特征重要性 (前20个):")
    indices = np.argsort(dt_model.feature_importances_)[::-1]
    for i in range(min(20, len(dt_model.feature_importances_))):
        print(f"{i+1}. {feature_names[indices[i]]}: {dt_model.feature_importances_[indices[i]]:.4f}")
    
    return dt_model, scaler, feature_names

def train_decision_tree_with_pca():
    """
    使用PCA降维和决策树训练模型
    """
    # 加载数据
    X, y, feature_names = load_and_preprocess_data()
    
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
    print(f"训练集标签分布: {np.bincount(y_train)}")
    print(f"测试集标签分布: {np.bincount(y_test)}")
    
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

    # 获取预测概率（用于ROC曲线）
    y_train_proba = dt_model.predict_proba(X_train_pca)[:, 1]
    y_test_proba = dt_model.predict_proba(X_test_pca)[:, 1]
    
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

    # 绘制ROC曲线
    auc_score = plot_roc_curve(y_test, y_test_proba, title="Decision Tree ROC Curve")
    print(f"测试集AUC值: {auc_score:.4f}")

    # 显示混淆矩阵
    plot_confusion_matrix(y_test, y_test_pred, title="Decision Tree Confusion Matrix")

    # 显示特征重要性 (注意：这是PCA变换后的特征重要性)
    plot_feature_importance(dt_model.feature_importances_, top_n=10, title="Top 10 Principal Components Importance")
    print("\n主成分重要性 (前10个):")
    indices = np.argsort(dt_model.feature_importances_)[::-1]
    for i in range(min(10, len(dt_model.feature_importances_))):
        print(f"{i+1}. PC {indices[i]}: {dt_model.feature_importances_[indices[i]]:.4f}")
    
    # 同时显示原始特征重要性（通过PCA逆变换近似）
    # 注意：这是一种近似的计算方式，不是完全精确的
    if hasattr(pca, 'components_'):
        # 将主成分重要性映射回原始特征
        pc_importances = dt_model.feature_importances_
        # 通过PCA组件的权重将重要性映射回原始特征
        original_feature_importances = np.abs(np.dot(pc_importances, pca.components_))
        
        plot_feature_importance(original_feature_importances, feature_names, top_n=20, 
                                title="Top 20 Original Features Importance (Mapped from PCA)")
        print("\n原始特征重要性 (从PCA映射，前20个):")
        orig_indices = np.argsort(original_feature_importances)[::-1]
        for i in range(min(20, len(original_feature_importances))):
            print(f"{i+1}. {feature_names[orig_indices[i]]}: {original_feature_importances[orig_indices[i]]:.4f}")
    
    return dt_model, pca, scaler


if __name__ == "__main__":
    # 先训练不带PCA的模型以获取原始特征重要性
    print("=" * 50)
    print("训练不带PCA的决策树模型")
    print("=" * 50)
    train_decision_tree_without_pca()
    
    print("\n" + "=" * 50)
    print("训练带PCA的决策树模型")
    print("=" * 50)
    train_decision_tree_with_pca()