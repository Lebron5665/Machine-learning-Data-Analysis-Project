import torch
import torch.nn as nn
import torchvision.models as models
try:
    import timm
except ImportError:
    print("错误: 请先安装 timm 库。运行: pip install timm")
    timm = None

def get_model(model_name, num_classes=10, pretrained=False):
    """
    获取模型实例。
    参数:
        model_name (str): 'resnet' 或 'vit'。
        num_classes (int): 分类数量 (CIFAR-10 为 10)。
        pretrained (bool): 是否使用预训练权重。
    """
    
    # ==========================================
    # 1. 构建 ResNet-18 
    # ==========================================
    if model_name.lower() == 'resnet':
        # 加载官方 ResNet18 结构
        net = models.resnet18(pretrained=pretrained)
        
        # 修改第一层卷积：从 7x7 stride=2 改为 3x3 stride=1
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除第一层后的 MaxPool，防止特征图过小
        net.maxpool = nn.Identity()
        
        # 修改全连接层 (FC Layer) 以匹配 CIFAR-10 的 10 类
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)
        
        return net

    # ==========================================
    # 2. 构建 Vision Transformer
    # ==========================================
    elif model_name.lower() == 'vit':
        # print(f"[Model] Initializing ViT-Tiny for CIFAR-10...")
        
        if timm is None:
            raise ImportError("请安装 timm: pip install timm")

        # 使用 timm 创建 ViT 模型
        net = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        return net
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'resnet' or 'vit'.")

if __name__ == "__main__":
    # 简单的测试代码
    try:
        resnet = get_model('resnet')
        print(f"ResNet OK. Params: {sum(p.numel() for p in resnet.parameters())}")
        
        vit = get_model('vit')
        print(f"ViT OK. Params: {sum(p.numel() for p in vit.parameters())}")
    except Exception as e:
        print(f"Error: {e}")
