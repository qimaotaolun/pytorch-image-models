# import timm # 0.7302350997924805 35 0.77
import torch
from torch.functional import F
from torch import nn
from einops import rearrange,repeat

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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
 
class MyModel(nn.Module):
    """Lightweight 3D CNN for multi-label classification (returns logits)."""

    def __init__(self, num_classes: int = 14, dim = 1024, pool = 'cls', depth = 1, heads = 8, dim_head = 64, mlp_dim = 2048, dropout = 0., emb_dropout = 0.):
        super(MyModel, self).__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
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
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        
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
        x = x.view(x.size(0), -1).unsqueeze(1)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
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
    model = MyModel(num_classes=14,depth=2)
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
