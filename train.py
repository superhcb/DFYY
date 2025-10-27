import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os
from tqdm import tqdm
from imblearn.over_sampling import SMOTE  # 添加SMOTE导入


from sklearn.metrics import precision_score, recall_score, confusion_matrix

# 导入模型
from model import Model

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

def train_model():
    """
    训练模型
    """
    # 加载数据
    X, y = load_and_preprocess_data()
    
    # 创建日志文件
    log_file_path = os.path.join('output', 'training_log.txt')
    log_file = open(log_file_path, 'a')
    
    def log_print(message):
        """同时打印到控制台和日志文件"""
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_print(f"数据形状: X={X.shape}, y={y.shape}")
    log_print(f"标签分布: {np.bincount(y)}")
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    log_print(f"训练集标签分布: {np.bincount(y_train)}")
    log_print(f"测试集标签分布: {np.bincount(y_test)}")

    # 应用SMOTE过采样到训练数据
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    log_print(f"SMOTE后训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    log_print(f"SMOTE后训练集标签分布: {np.bincount(y_train)}")

    # # 计算类别权重以处理类别不平衡
    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # class_weights = torch.FloatTensor(class_weights)
    
    # log_print(f"类别权重: {class_weights}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 初始化模型
    model = Model()
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch_X, batch_y in train_bar:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每10个epoch打印一次训练信息
        # if (epoch + 1) % 10 == 0:
            # 计算训练准确率
        model.eval()
        with torch.no_grad():
            # 训练集指标
            train_outputs = model(X_train_tensor)
            _, train_predicted = torch.max(train_outputs.data, 1)
            train_accuracy = (train_predicted == y_train_tensor).sum().item() / len(y_train_tensor)
            
            # 测试集指标
            test_outputs = model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            
            # 计算详细的测试集指标
            test_precision, test_sensitivity, test_specificity = calculate_metrics(
                y_test_tensor.numpy(), test_predicted.numpy()
            )
        
        log_print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, '
                  f'Test Precision: {test_precision:.4f}, Test Sensitivity: {test_sensitivity:.4f}, '
                  f'Test Specificity: {test_specificity:.4f}')
    

    # 最终评估
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        
        # 计算详细指标
        precision, sensitivity, specificity = calculate_metrics(
            y_test_tensor.numpy(), predicted.numpy()
        )
        
        log_print(f'最终测试准确率: {accuracy:.4f}, 精确率: {precision:.4f}, '
                  f'敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}')
    
    # 关闭日志文件
    log_file.close()
    
    # 保存模型
    model_path = os.path.join('output', 'coronary_stenosis_model.pth')
    torch.save(model.state_dict(), model_path)
    print("模型已保存为 output/coronary_stenosis_model.pth")
    print("训练日志已保存为 output/training_log.txt")
    
    return model

if __name__ == '__main__':
    # 检查是否有CUDA可用
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 训练模型
    model = train_model()