import torch
from torch import nn
import timm
from timm.layers import NormMlpClassifierHead
from typing import List, Optional


class Conv3DStageFusion(nn.Module):
    """
    分阶段 3D 分支，用于在 ConvNeXt 每个阶段后进行特征融合（Affine/FiLM）。
    - 输入: (B, 1, D, H, W)
    - 经过 depth_3d 个 Conv3d-BN-ReLU-MaxPool3d 块；每个块后:
        * 自适应池化到 (p, p, p)
        * 展平 + 线性投影到 embed_dim_3d
        * 经线性映射生成与 2D 阶段通道数匹配的 (gamma, beta) 用于 FiLM 融合
    """
    def __init__(
        self,
        stage_channels: List[int],       # ConvNeXt 四个阶段的通道数，如 [192, 384, 768, 1536]
        in_chans_3d: int = 1,
        depth_3d: Optional[int] = None,  # 默认为 len(stage_channels)
        base_ch: int = 16,
        max_ch: int = 256,
        pool_out: int = 2,
        embed_dim_3d: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stage_channels = stage_channels
        self.depth_3d = depth_3d or len(stage_channels)
        assert self.depth_3d == len(stage_channels), "3D 分支深度需与 2D 分支阶段数一致，以便逐阶段融合。"

        # 3D 块序列
        self.blocks = nn.ModuleList()
        in_c = in_chans_3d
        self.out_cs: List[int] = []
        for i in range(self.depth_3d):
            out_c = min(max_ch, base_ch * (2 ** i))
            blk = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),
            )
            self.blocks.append(blk)
            self.out_cs.append(out_c)
            in_c = out_c

        self.pool = nn.AdaptiveAvgPool3d((pool_out, pool_out, pool_out))

        # 每阶段：投影到 embed_dim_3d
        self.stage_proj = nn.ModuleList()
        for i in range(self.depth_3d):
            flat_dim = self.out_cs[i] * (pool_out ** 3)
            self.stage_proj.append(
                nn.Sequential(
                    nn.Linear(flat_dim, embed_dim_3d),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )

        # 每阶段：从 embed_dim_3d 生成 gamma/beta 映射
        self.affine_mappers = nn.ModuleList()
        for i in range(self.depth_3d):
            c2d = self.stage_channels[i]
            self.affine_mappers.append(nn.Linear(embed_dim_3d, 2 * c2d))

        # 记录最后阶段的嵌入（用于最终融合）
        self.last_embed: Optional[torch.Tensor] = None

    def forward_stage(self, x3d: torch.Tensor, stage_idx: int):
        """
        前向单阶段:
        - 应用第 stage_idx 个 3D 块
        - 生成 embed, 再映射为 (gamma, beta)
        返回: gamma (B, C_i), beta (B, C_i), x3d (更新后的特征), embed (B, embed_dim_3d)
        """
        x3d = self.blocks[stage_idx](x3d)
        pooled = self.pool(x3d)
        flat = pooled.view(pooled.size(0), -1)
        embed = self.stage_proj[stage_idx](flat)
        affine = self.affine_mappers[stage_idx](embed)
        c2d = self.stage_channels[stage_idx]
        gamma, beta = affine[:, :c2d], affine[:, c2d:]
        if stage_idx == self.depth_3d - 1:
            self.last_embed = embed
        return gamma, beta, x3d, embed


class MyModel(nn.Module):
    """
    分阶段融合版：ConvNeXt-Large DINOv3 2D 分支 + 3D 分支
    - 在 ConvNeXt 四个阶段后，使用 3D 分支生成的 (gamma, beta) 对 2D 特征做通道级 FiLM 融合
      x2d <- x2d * (1 + alpha * tanh(gamma)) + beta
    - 最终取 ConvNeXt pre_logits 向量与 3D 最终嵌入向量拼接，经过分类头输出 logits（BCEWithLogits）
    """
    def __init__(
        self,
        num_classes: int = 14,
        in_chans: int = 32,                          # 切片数作为 2D 分支通道数
        pretrained: bool = False,
        backbone_name: str = "convnext_large.dinov3_lvd1689m",
        # 2D/3D 末端融合投影维度
        embed_dim_2d: int = 512,
        embed_dim_3d: int = 512,
        # 3D 分支结构
        depth_3d: Optional[int] = None,              # 默认为 ConvNeXt 阶段数（通常为 4）
        base_ch_3d: int = 16,
        max_ch_3d: int = 256,
        pool_out_3d: int = 2,
        dropout_3d: float = 0.1,
        # 分阶段融合强度
        fusion_alpha: float = 0.2,                   # FiLM 中 tanh(gamma) 的缩放系数
        fusion_dropout: float = 0.2,
    ):
        super().__init__()
        # 2D 主干
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,        # 输出特征而非分类结果
            in_chans=in_chans,
        )

        # 读取 2D 阶段通道数
        assert hasattr(self.backbone, "feature_info"), "ConvNeXt 主干缺少 feature_info。"
        self.stage_channels: List[int] = [fi["num_chs"] for fi in self.backbone.feature_info]
        assert len(self.stage_channels) == 4, f"期望 4 个阶段，得到 {len(self.stage_channels)}。"

        # 3D 分支（分阶段）
        self.branch3d = Conv3DStageFusion(
            stage_channels=self.stage_channels,
            in_chans_3d=1,
            depth_3d=depth_3d or len(self.stage_channels),
            base_ch=base_ch_3d,
            max_ch=max_ch_3d,
            pool_out=pool_out_3d,
            embed_dim_3d=embed_dim_3d,
            dropout=dropout_3d,
        )

        # 2D pre_logits -> embed_dim_2d
        # 从 ConvNeXt 取 pre_logits，先 LayerNorm 再线性映射
        feat2d_dim = getattr(self.backbone, "head_hidden_size", None)
        if not feat2d_dim or feat2d_dim == 0:
            feat2d_dim = getattr(self.backbone, "num_features", None)
        if not feat2d_dim:
            raise RuntimeError("无法确定 ConvNeXt 主干的特征维度（head_hidden_size / num_features 均不可用）")
        self.proj2d = nn.Sequential(
            nn.LayerNorm(feat2d_dim),
            nn.Linear(feat2d_dim, embed_dim_2d),
            nn.ReLU(inplace=True),
        )

        # 末端分类头（拼接 2D/3D 向量），沿用 timm 的 NormMlpClassifierHead
        fusion_dim = embed_dim_2d + embed_dim_3d
        self.classifier = NormMlpClassifierHead(
            in_features=fusion_dim,
            num_classes=num_classes,
            hidden_size=None,
            pool_type='avg',              # 保持与 ConvNeXt 默认一致：池化->归一化->MLP
            drop_rate=fusion_dropout,
            norm_layer='layernorm2d',
            act_layer='gelu',             # 与 ConvNeXt 的 head act 对齐
        )

        self.fusion_alpha = fusion_alpha

    def _film(self, x2d: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        通道级 FiLM 融合：
        y = x * (1 + alpha * tanh(gamma)) + beta
        """
        scale = 1.0 + torch.tanh(gamma) * self.fusion_alpha
        y = x2d * scale.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向：
        - 输入 x: (B, C, H, W)，其中 C=切片数（例如 32）
        - 输出 logits: (B, num_classes)
        """
        assert x.dim() == 4, f"期望输入形状为 (B, C, H, W)，得到 {tuple(x.shape)}"

        # 2D 干部分阶段前向 + 融合
        x2d = self.backbone.stem(x)
        x3d = x.unsqueeze(1)  # (B, 1, D(=C), H, W)
        for i, stage in enumerate(self.backbone.stages):
            x2d = stage(x2d)
            gamma, beta, x3d, _embed_i = self.branch3d.forward_stage(x3d, i)
            x2d = self._film(x2d, gamma, beta)

        # 2D 末端归一化 + 提取 pre_logits
        x2d = self.backbone.norm_pre(x2d)
        z2d = self.backbone.forward_head(x2d, pre_logits=True)  # (B, feat2d_dim)
        z2d = self.proj2d(z2d)                                  # (B, embed_dim_2d)

        # 3D 最终嵌入
        z3d = self.branch3d.last_embed
        assert z3d is not None, "3D 分支最后嵌入不存在，请检查 forward_stage 调用。"

        # 末端融合与分类（logits）
        z = torch.cat([z2d, z3d], dim=1)          # (B, fusion_dim)
        z_nchw = z.unsqueeze(-1).unsqueeze(-1)    # 适配 NormMlpClassifierHead 的 NCHW 输入 (B, C, 1, 1)
        logits = self.classifier(z_nchw)          # (B, num_classes)
        return logits

    # ===== 训练阶段的可选冻结接口 =====
    def freeze_backbone(self, train_norm: bool = False):
        """
        冻结 2D 主干参数；若 train_norm=True，保留正则化层参数可训练。
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        if train_norm:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for p in m.parameters(recurse=False):
                        p.requires_grad = True

    def unfreeze_backbone(self):
        """解冻 2D 主干参数。"""
        for p in self.backbone.parameters():
            p.requires_grad = True

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


