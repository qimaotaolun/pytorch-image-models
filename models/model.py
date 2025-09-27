import timm
import torch
from torch.functional import F
from torch import nn

# class MyModel(torch.nn.Module):
#     def __init__(self, num_classes=1):
#         super(MyModel, self).__init__()
#         print(f"Initializing model...{num_classes}")
#         self.timm_0 = timm.create_model(
#             "tf_efficientnetv2_s.in21k_ft_in1k", 
#             num_classes=num_classes, 
#             pretrained=True,  # Don't load pretrained weights
#             in_chans=32
#         )
        
#     def forward(self, x):
#         x = self.timm_0(x)
#         return x
    
class MyModel(nn.Module):
    """Lightweight 3D CNN for multi-label classification (returns logits)."""

    def __init__(self, num_classes: int = 14):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.fc1 = nn.Linear(128 * 2 * 2 * 2, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B,1,D,H,W)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout1(x)
        x = F.relu(self.fc2(x)); x = self.dropout2(x)
        x = self.fc3(x)  # logits
        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印当前GPU的使用情况
    if device.type == 'cuda':
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    else:
        print("CUDA not available, using CPU.")

    # Initialize model and move to appropriate device
    model = MyModel(num_classes=14)
    model = model.to(device)

    # Load the saved model weights
    # model.load_state_dict(torch.load('model_best.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Check memory usage before forward pass
    if device.type == 'cuda':
        print(f"Allocated Memory before operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory before operation: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    
    # Create a dummy image tensor
    image = torch.randn(1, 1, 32, 384, 384).to(device)

    # Run forward pass
    output = model(image)
    
    # Apply sigmoid activation
    print(F.sigmoid(output))

    # Check and print GPU memory usage after the operation
    if device.type == 'cuda':
        print(f"Allocated Memory after operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory after operation: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
