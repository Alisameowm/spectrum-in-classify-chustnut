# 添加训练所需的库
import torch
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def train_model(model, train_loader, val_loader, test_loader, 
                criterion, optimizer, scheduler=None, num_epochs=50, 
                device=None, patience=100):
    """
    完整的模型训练、验证和测试函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器(可选)
        num_epochs: 训练轮数
        device: 计算设备
        patience: 早停耐心值
    
    返回:
        训练历史记录字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化记录列表
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # 记录最佳模型和性能
    best_model_wts = model.state_dict()
    best_acc = 0.0
    no_improve_epochs = 0
    
    print(f"Training started, using device: {device}")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 30)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for features, labels in train_bar:
            features, labels = features.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })
        
        # 如果有学习率调度器，更新学习率
        if scheduler is not None:
            scheduler.step()
            
        # 计算平均损失和准确率
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # 记录训练指标
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation", leave=False)
            for features, labels in val_bar:
                features, labels = features.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': val_correct / val_total
                })
        
        # 计算平均损失和准确率
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # 记录验证指标
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # 打印本轮指标
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # 检查并保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model.state_dict()
            no_improve_epochs = 0
            print(f"New best model! Validation accuracy: {best_acc:.4f}")
        else:
            no_improve_epochs += 1
        
        # 早停检查
        if no_improve_epochs >= patience:
            print(f"Validation performance has not improved for {patience} epochs, early stopping")
            break
            
        print()
    
    # 训练结束统计
    time_elapsed = time.time() - start_time
    print(f"Training complete, took {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    
    # 绘制训练曲线
    plot_training_history(history)
    
    # 测试阶段
    test_metrics = test_model(model, test_loader, criterion, device)
    
    # 返回训练历史和最佳模型
    return history, model

def test_model(model, test_loader, criterion, device):
    """测试模型性能"""
    print("\n开始测试最终模型...")
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # 收集预测和真实标签用于混淆矩阵
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        for features, labels in test_bar:
            features, labels = features.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 统计
            test_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # 收集预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            test_bar.set_postfix({
                'loss': loss.item(),
                'acc': test_correct / test_total
            })
    
    # 计算最终指标
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    print(f"\nTest results:")
    print(f"Loss: {test_loss:.4f} Accuracy: {test_acc:.4f}")
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion matrix:")
    print(cm)
    
    # 生成分类报告
    class_names = ["Type 1", "Type 2"]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification report:")
    print(report)
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'confusion_matrix': cm,
        'report': report
    }

def plot_training_history(history):
    """绘制训练历史曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
