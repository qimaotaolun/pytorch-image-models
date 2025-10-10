# import timm
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

    def __init__(self, num_classes: int = 14, dim: int = 1024, pool: str = 'cls', depth: int = 4, tranformer_depth: int = 4, heads = 8, dim_head = 64, mlp_dim = 2048, transformer_dropout = 0., emb_dropout = 0.):
        super(MyModel, self).__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.depth = depth
        self.tranfomer_depth = tranformer_depth

        # 使用 depth 构建 CNN 块（ModuleList 包含 Conv3d、BN、ReLU、MaxPool3d 四者的顺序模块）
        self.cnn_blocks = nn.ModuleList()
        in_channels = 1
        for i in range(self.depth):
            out_channels = min(512, 32 * (2 ** i))
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2)
            )
            self.cnn_blocks.append(block)
            in_channels = out_channels
    
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        # CNN 输出通道与展平维度
        self.cnn_out_channels = in_channels
        self.flatten_dim = self.cnn_out_channels * 2 * 2 * 2
        
        if self.tranfomer_depth > 0: # Transformer
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, 2, dim))
            self.dropout = nn.Dropout(emb_dropout)

            # 若 CNN 展平维度与 Transformer 预期 dim 不一致，添加线性投影
            self.pre_transformer_proj = nn.Identity() if self.flatten_dim == dim else nn.Linear(self.flatten_dim, dim)

            self.transformer = Transformer(dim, self.tranfomer_depth, heads, dim_head, mlp_dim, transformer_dropout)

            self.pool = pool
        
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def freeze_transformer(self):
        """
        冻结 Transformer 子模块（仅在 self.tranfomer_depth > 0 时存在）以及 CLS/位置参数的梯度。
        """
        if self.tranfomer_depth > 0:
            for p in self.transformer.parameters():
                p.requires_grad = False
            # cls_token 与 pos_embedding 是 nn.Parameter
            self.cls_token.requires_grad = False
            self.pos_embedding.requires_grad = False

    def unfreeze_transformer(self):
        """
        解冻 Transformer 子模块（仅在 self.tranfomer_depth > 0 时存在）以及 CLS/位置参数的梯度。
        """
        if self.tranfomer_depth > 0:
            for p in self.transformer.parameters():
                p.requires_grad = True
            self.cls_token.requires_grad = True
            self.pos_embedding.requires_grad = True

    def freeze_cnn_classifier(self):
        """
        冻结卷积/BN 以及分类头（全连接层）的梯度。
        说明：池化层与 Dropout 没有可训练参数，忽略即可。
        """
        modules = [*self.cnn_blocks, self.fc1, self.fc2, self.fc3]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_cnn_classifier(self):
        """
        解冻卷积/BN 以及分类头（全连接层）的梯度。
        """
        modules = [*self.cnn_blocks, self.fc1, self.fc2, self.fc3]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = True

    def save_cnn_classifier_weights(self, path: str) -> None:
        """
        仅保存 CNN + 分类头（fc1, fc2, fc3）权重到文件。
        保存为 state_dict(dict)，包含 BN 的 running stats 等缓冲区；不包含 Transformer 相关参数。
        """
        container = nn.ModuleDict({
            'cnn_blocks': self.cnn_blocks,
            'fc1': self.fc1,
            'fc2': self.fc2,
            'fc3': self.fc3,
        })
        state = container.state_dict()
        # 将张量转为 CPU，避免跨设备问题
        state_cpu = {k: v.detach().cpu() for k, v in state.items()}
        torch.save(state_cpu, path)

    def load_cnn_classifier_weights(self, path: str, strict: bool = True) -> None:
        """
        从文件加载 CNN + 分类头（fc1, fc2, fc3）权重。
        - strict: 是否严格匹配键名。
        - 自动兼容常见前缀（如 DataParallel 的 'module.' 或外层 'model.'）。
        """
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']

        # 去除常见的前缀，避免键名不一致
        if any(k.startswith('module.') for k in sd.keys()):
            sd = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
        if any(k.startswith('model.') for k in sd.keys()):
            sd = { (k[6:] if k.startswith('model.') else k): v for k, v in sd.items() }

        container = nn.ModuleDict({
            'cnn_blocks': self.cnn_blocks,
            'fc1': self.fc1,
            'fc2': self.fc2,
            'fc3': self.fc3,
        })
        container.load_state_dict(sd, strict=strict)
    def forward(self, x):
        # x: (B,1,D,H,W)
        # 根据 depth 用 for 遍历 CNN 块（ModuleList 包含 Conv+BN+ReLU+Pool）
        for block in self.cnn_blocks:
            x = block(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        if self.tranfomer_depth > 0: # Transformer
            x = self.pre_transformer_proj(x)
            x = x.unsqueeze(1)
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
    model = MyModel(num_classes=14,depth=5,tranformer_depth=0,transformer_dropout=0.)
    # sd = torch.load('model_best.pth')
    # if any(k.startswith('module.') for k in sd.keys()):
    #     sd = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
    # model.load_state_dict(sd)
    
    # model.freeze_cnn_classifier()
    # model.freeze_transformer()
    # 在 main 中输出当前可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,}")
    
    # model.unfreeze_cnn_classifier()
    # model.unfreeze_transformer()
    
    # model.save_cnn_classifier_weights('cnn_classifier_weights.pth')
    # model.load_cnn_classifier_weights('cnn_classifier_weights.pth',strict=True)
    
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

