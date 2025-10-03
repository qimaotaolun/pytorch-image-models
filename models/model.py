import torch
from torch import nn
from torch.nn import functional as F


class SqueezeExcitation3d(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps.
    """
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        reduced_chs = max(8, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, reduced_chs, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv3d(reduced_chs, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)
        return x * s


class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath for residual connections.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with broadcast over batch, 1D mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor


class MBConv3d(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (3D variant) with optional Squeeze-and-Excitation.
    - Expand (1x1x1)
    - Depthwise (3x3x3)
    - SE
    - Project (1x1x1)
    - Residual if shape & stride match
    """
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: tuple[int, int, int] = (1, 1, 1),
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.BatchNorm3d,
        act_layer: nn.Module = nn.SiLU,
    ):
        super().__init__()
        assert stride in {(1, 1, 1), (1, 2, 2)}, "Only spatial downsample stride (1,2,2) or identity (1,1,1) supported."
        self.use_residual = stride == (1, 1, 1) and in_chs == out_chs
        mid_chs = in_chs * expand_ratio

        layers = []
        # expand
        if expand_ratio != 1:
            layers += [
                nn.Conv3d(in_chs, mid_chs, kernel_size=1, bias=False),
                norm_layer(mid_chs),
                act_layer(inplace=True),
            ]
        else:
            mid_chs = in_chs

        # depthwise 3x3x3 (grouped by channels)
        layers += [
            nn.Conv3d(mid_chs, mid_chs, kernel_size=3, stride=stride, padding=1, groups=mid_chs, bias=False),
            norm_layer(mid_chs),
            act_layer(inplace=True),
        ]

        # SE
        self.se = SqueezeExcitation3d(mid_chs, se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()

        # project
        layers += [
            nn.Conv3d(mid_chs, out_chs, kernel_size=1, bias=False),
            norm_layer(out_chs),
        ]

        self.blocks = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        # expand + depthwise
        out = self.blocks[0](out) if isinstance(self.blocks[0], nn.Conv3d) else out  # optional expand conv
        start_idx = 0 if isinstance(self.blocks[0], nn.Conv3d) else -1  # track index shift

        # If expand, the sequence is: [conv1x1, bn, act, dwconv, bn, act, ...]
        # If no expand, we start at depthwise conv in sequence index 0.
        if start_idx == 0:
            out = self.blocks[1](out)  # bn
            out = self.blocks[2](out)  # act
            dw_idx = 3
        else:
            dw_idx = 0

        out = self.blocks[dw_idx](out)      # dwconv
        out = self.blocks[dw_idx + 1](out)  # bn
        out = self.blocks[dw_idx + 2](out)  # act

        out = self.se(out)

        proj_idx = dw_idx + 3
        out = self.blocks[proj_idx](out)      # proj conv
        out = self.blocks[proj_idx + 1](out)  # bn

        if self.use_residual:
            out = self.drop_path(out)
            out = out + x
        return out


class EfficientNetV2_3D(nn.Module):
    """
    A lightweight 3D adaptation of EfficientNetV2-S for volumetric medical imaging.

    Design choices:
    - Only spatial downsampling (stride=(1,2,2)) to preserve depth axis (D).
    - 3D MBConv blocks with SE.
    - Global average pooling over all axes before classification.
    """
    def __init__(
        self,
        in_chs: int = 1,
        num_classes: int = 14,
        widths: tuple[int, ...] = (24, 48, 80, 160, 256),
        repeats: tuple[int, ...] = (2, 4, 4, 6, 6),
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm3d,
        act_layer: nn.Module = nn.SiLU,
        head_chs: int = 512,
    ):
        super().__init__()
        assert len(widths) == len(repeats), "widths and repeats must align"
        # Stem (downsample spatial only)
        self.stem = nn.Sequential(
            nn.Conv3d(in_chs, widths[0], kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            norm_layer(widths[0]),
            act_layer(inplace=True),
        )

        # Stages
        stages = []
        total_blocks = sum(repeats)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        in_c = widths[0]
        for stage_idx, (out_c, n_rep) in enumerate(zip(widths, repeats)):
            for rep in range(n_rep):
                stride = (1, 2, 2) if rep == 0 and stage_idx > 0 else (1, 1, 1)
                stages.append(
                    MBConv3d(
                        in_chs=in_c,
                        out_chs=out_c,
                        stride=stride,
                        expand_ratio=expand_ratio,
                        se_ratio=se_ratio,
                        drop_path=dp_rates[dp_idx],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                )
                dp_idx += 1
                in_c = out_c
        self.blocks = nn.Sequential(*stages)

        # Head
        self.head = nn.Sequential(
            nn.Conv3d(in_c, head_chs, kernel_size=1, bias=False),
            norm_layer(head_chs),
            act_layer(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(head_chs, num_classes)

        # Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class MyModel(nn.Module):
    """
    Wrapper model exposing the same interface used by the training script,
    backed by EfficientNetV2_3D.

    Arguments depth / tranformer_depth / transformer_dropout are accepted for compatibility,
    but currently unused by this 3D EfficientNetV2 backbone.
    """
    def __init__(
        self,
        num_classes: int = 14,
        depth: int | None = None,
        tranformer_depth: int | None = None,
        transformer_dropout: float | None = None,
        in_chans: int = 1,
    ):
        super().__init__()
        self.backbone = EfficientNetV2_3D(in_chs=in_chans, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # Compatibility helpers (no-op or broad freeze)
    def freeze_cnn_classifier(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_cnn_classifier(self):
        for p in self.parameters():
            p.requires_grad = True

    def save_cnn_classifier_weights(self, path: str) -> None:
        state_cpu = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        torch.save(state_cpu, path)

    def load_cnn_classifier_weights(self, path: str, strict: bool = True) -> None:
        try:
            sd = torch.load(path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
                sd = sd['state_dict']
            # strip common prefixes
            if any(k.startswith('module.') for k in sd.keys()):
                sd = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
            if any(k.startswith('model.') for k in sd.keys()):
                sd = { (k[6:] if k.startswith('model.') else k): v for k, v in sd.items() }
            self.load_state_dict(sd, strict=strict)
        except Exception as e:
            print(f"Warning: load_cnn_classifier_weights failed: {e}")


if __name__ == '__main__':
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    else:
        print("CUDA not available, using CPU.")

    model = MyModel(num_classes=14, in_chans=1)
    model = model.to(device)
    model.eval()

    if device.type == 'cuda':
        print(f"Allocated Memory before operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory before operation: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")

    # Dummy 3D input: (B, C=1, D=32, H=384, W=384)
    image = torch.randn(1, 1, 32, 384, 384, device=device)
    with torch.inference_mode():
        output = model(image)
        print(torch.sigmoid(output))

    if device.type == 'cuda':
        print(f"Allocated Memory after operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory after operation: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
