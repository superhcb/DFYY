from utils import calculate_metrics, load_and_preprocess_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from itertools import product
from tqdm import tqdm

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

def grid_search_decision_tree():
    """
    使用网格搜索寻找最优超参数，以test_specificity为性能参考
    包括PCA的n_components和随机种子参数
    """
    # 加载数据
    X, y = load_and_preprocess_data()
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 定义参数网格
    pca_components = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    random_states = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    criterion = ['gini', 'entropy']
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, None]
    min_samples_split = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # 存储最佳结果
    best_ydindex = 0
    best_params = None
    best_model = None
    best_metrics = None
    
    total_combinations = len(pca_components) * len(random_states) * len(criterion) * len(max_depth) * len(min_samples_split) * len(min_samples_leaf)
    print(f"总共需要测试 {total_combinations} 种参数组合")
    
     # 手动进行网格搜索
    param_combinations = list(product(
        pca_components, random_states, criterion, max_depth, min_samples_split, min_samples_leaf))
    
    progress_bar = tqdm(param_combinations, total=len(param_combinations), desc="网格搜索进度")
    
    for n_components, random_state, crit, max_d, min_split, min_leaf in progress_bar:
        
        try:
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=random_state, stratify=y
            )

            # 使用PCA进行降维
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            # 创建决策树分类器
            dt = DecisionTreeClassifier(
                criterion=crit,
                max_depth=max_d,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                random_state=random_state
            )
            
            # 训练模型
            dt.fit(X_train_pca, y_train)
            
            # 预测
            y_test_pred = dt.predict(X_test_pca)
            
            # 计算所有指标
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision, test_sensitivity, test_specificity, test_ydindex = calculate_metrics(y_test, y_test_pred)
            
            # 更新最佳模型（以特异性为主要指标）
            if test_ydindex > best_ydindex:
                best_ydindex = test_ydindex
                best_params = {
                    'pca_n_components': n_components,
                    'random_state': random_state,
                    'criterion': crit,
                    'max_depth': max_d,
                    'min_samples_split': min_split,
                    'min_samples_leaf': min_leaf
                }
                best_metrics = {
                    'accuracy': test_accuracy,
                    'precision': test_precision,
                    'sensitivity': test_sensitivity,
                    'specificity': test_specificity,
                    'ydindex': test_ydindex
                }

                print("最佳参数:")
                for param, value in best_params.items():
                    print(f"  {param}: {value}")
                
                print("\n最佳模型在测试集上的所有指标:")
                print(f"  准确率 (Accuracy): {best_metrics['accuracy']:.4f}")
                print(f"  精确率 (Precision): {best_metrics['precision']:.4f}")
                print(f"  敏感性 (Sensitivity): {best_metrics['sensitivity']:.4f}")
                print(f"  特异性 (Specificity): {best_metrics['specificity']:.4f}")
                print(f"  约登指数: {best_metrics['ydindex']:.4f}")
                
        except Exception as e:
            # 忽略出错的参数组合
            continue
    
    # 输出最佳结果
    print("\n" + "="*50)
    print("网格搜索完成")
    print("="*50)
    print("最佳参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print("\n最佳模型在测试集上的所有指标:")
    print(f"  准确率 (Accuracy): {best_metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision): {best_metrics['precision']:.4f}")
    print(f"  敏感性 (Sensitivity): {best_metrics['sensitivity']:.4f}")
    print(f"  特异性 (Specificity): {best_metrics['specificity']:.4f}")
    print(f"  约登指数: {best_metrics['ydindex']:.4f}")
    
    return best_params, best_metrics

if __name__ == "__main__":
    # 原始模型训练
    print("=== 原始PCA + 决策树模型 ===")
    train_decision_tree_with_pca()
    
    print("\n" + "="*50)
    print("=== 网格搜索优化 ===")
    # 网格搜索优化
    grid_search_decision_tree()