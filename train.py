import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os

# 导入我们之前写的模块
from dataset import get_data_loaders
from models import get_model

# -----------------------------------------------------------
# 1. 配置参数 (Hyperparameters)
# -----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64     # 如果显存不够 (如 ViT OOM)，请调小到 32
LEARNING_RATE = 1e-4 # 对于 ViT 和 ResNet 通用的较小学习率
EPOCHS = 30        
DATA_DIR = './data'

print(f"Using device: {DEVICE}")

# -----------------------------------------------------------
# 2. 定义训练和评估函数
# -----------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    loss = running_loss / total
    acc = 100. * correct / total
    return loss, acc

# -----------------------------------------------------------
# 3. 主运行逻辑 (Main Loop)
# -----------------------------------------------------------
def run_experiment(model_name):
    print(f"\n{'='*20} Start Training {model_name.upper()} {'='*20}")
    
    # 1. 获取数据
    train_loader, test_loader = get_data_loaders(model_name, BATCH_SIZE, DATA_DIR)
    
    # 2. 构建模型
    model = get_model(model_name, num_classes=10, pretrained=False).to(DEVICE)
    
    # 3. 定义优化器和损失
    # 注意：ViT 通常推荐使用 AdamW，为了公平对比，这里两者都用 AdamW
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 记录历史数据用于绘图
    history = {'train_loss': [], 'test_acc': [], 'train_time': 0}
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = evaluate(model, test_loader, criterion)
        
        history['train_loss'].append(t_loss)
        history['test_acc'].append(v_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | Test Acc: {v_acc:.2f}%")
        
    total_time = time.time() - start_time
    history['train_time'] = total_time
    print(f"Finished {model_name}. Total Time: {total_time:.2f}s")
    
    return history

if __name__ == "__main__":
    # -------------------------------------------------------
    # 运行实验
    # -------------------------------------------------------
    # 运行 ResNet
    resnet_history = run_experiment('resnet')
    
    # 运行 ViT (注意：如果没有 GPU，ViT 可能会比较慢)
    vit_history = run_experiment('vit')
    
    # -------------------------------------------------------
    # 4. 绘图与分析 (Analysis Layer)
    # -------------------------------------------------------
    print("\nPlotting results...")
    epochs_range = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 子图 1: 训练 Loss 对比
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, resnet_history['train_loss'], label='ResNet-18', marker='o')
    plt.plot(epochs_range, vit_history['train_loss'], label='ViT-Tiny', marker='s')
    plt.title('Training Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 子图 2: 测试集准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, resnet_history['test_acc'], label='ResNet-18', marker='o')
    plt.plot(epochs_range, vit_history['test_acc'], label='ViT-Tiny', marker='s')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 保存图片到本地，用于写报告
    save_path = 'comparison_result.png'
    plt.savefig(save_path)
    print(f"Result plot saved to {save_path}")
    plt.show()
    
    # 打印最终总结
    print(f"\n{'='*10} Final Report Data {'='*10}")
    print(f"ResNet-18 -> Time: {resnet_history['train_time']:.1f}s | Best Acc: {max(resnet_history['test_acc']):.2f}%")
    print(f"ViT-Tiny  -> Time: {vit_history['train_time']:.1f}s | Best Acc: {max(vit_history['test_acc']):.2f}%")