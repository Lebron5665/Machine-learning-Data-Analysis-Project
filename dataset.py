import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(model_name, batch_size=64, data_dir='./data', num_workers=2):
    """
    根据模型类型加载 CIFAR-10 数据集并返回 DataLoaders。
    
    参数:
        model_name (str): 'resnet' 或 'vit'。
                          - 'resnet': 使用 32x32 分辨率。
                          - 'vit': 使用 Resize((224, 224)) 以适应 Transformer Patch 切分。
        batch_size (int): 批次大小，默认为 64。
        data_dir (str): 数据下载/存放路径。
        num_workers (int): 数据加载线程数。
    
    返回:
        trainloader, testloader
    """
    
    # CIFAR-10 的官方均值和方差
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # -----------------------------------------------------------
    # 1. 定义预处理 (Transforms)
    # -----------------------------------------------------------
    if model_name.lower() == 'vit':
        # Vision Transformer 需要较大的分辨率 (通常 224) 才能切分出足够的 Patch
        # 如果是 ViT，我们强制将图片放大
        print(f"[Data] Mode: ViT. Resizing images to 224x224.")
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 关键步骤：上采样
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        
    elif model_name.lower() == 'resnet':
        # ResNet 可以直接处理 32x32 的小图，这是 CNN 的优势（归纳偏置）
        print(f"[Data] Mode: ResNet. Using native 32x32 resolution.")
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # 典型的 CIFAR 增强
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
    else:
        raise ValueError("model_name must be either 'resnet' or 'vit'")

    # -----------------------------------------------------------
    # 2. 下载并加载数据集
    # -----------------------------------------------------------
    print(f"[Data] Downloading/Loading CIFAR-10 to {data_dir}...")
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )

    # -----------------------------------------------------------
    # 3. 创建 DataLoader
    # -----------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    print(f"[Data] Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    
    return train_loader, test_loader

# 简单的测试代码，确保可以直接运行检查
if __name__ == "__main__":
    # 测试获取 ResNet 的数据加载器
    train_loader, _ = get_data_loaders(model_name='resnet', batch_size=4)
    images, labels = next(iter(train_loader))
    print(f"ResNet Batch Shape: {images.shape}") # 应为 [4, 3, 32, 32]

    # 测试获取 ViT 的数据加载器
    train_loader, _ = get_data_loaders(model_name='vit', batch_size=4)
    images, labels = next(iter(train_loader))
    print(f"ViT Batch Shape: {images.shape}")    # 应为 [4, 3, 224, 224]