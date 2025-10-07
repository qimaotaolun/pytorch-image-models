import torch
from torch import nn
import timm
from typing import Tuple, Literal

class Conv3DStemTo2D(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 2, 2),
        padding: Tuple[int, int, int] = (1, 1, 1),
        bias: bool = False,
        reduce: Literal['mean', 'max'] = 'mean',
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_chans,
            out_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        y = self.conv3d(x)  # (B, out_chans, D', H', W')
        if self.reduce == 'mean':
            y = y.mean(dim=2)  # reduce depth to get 2D feature map
        else:
            y = y.max(dim=2).values
        return y  # (B, out_chans, H', W')

def create_tf_efficientnetv2_s_3d(
    num_classes: int = 14,
    in_chans: int = 1,
    pretrained: bool = False,
    kd: int = 3,
    reduce: Literal['mean', 'max'] = 'mean',
    stem_in_chans_2d: int = 32,
) -> nn.Module:
    """
    创建3D输入版本的 tf_efficientnetv2_s.in21k_ft_in1k，仅替换第一层为3D卷积，其他保持不变。
    - 输入形状: (B, in_chans, D, H, W)
    - 第一层: Conv3d with stride (1,2,2), kernel (kd,3,3), padding (1,1,1)，再沿深度维做 reduce('mean'/'max') 得到2D特征图供后续2D网络使用
    - 与原始代码对应的改动入口为 timm 的创建语句：model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', num_classes=14, pretrained=False, in_chans=32)，其中 in_chans=32 仅用于原始 2D stem 的配置（会被替换为 3D stem），其余网络保持一致。
    """
    # 构建原始2D模型（保持除stem外的所有权重）
    model = timm.create_model(
        'tf_efficientnetv2_s.in21k_ft_in1k',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=stem_in_chans_2d,  # 与用户给出的入口一致，随后替换 stem 为 3D
    )
    # 读取原始stem的输出通道数以保持兼容
    out_chans = getattr(model.conv_stem, 'out_channels', None)
    if out_chans is None:
        # 兼容 Conv2dSame / 自定义模块，尝试从 weight 形状推断
        weight = getattr(model.conv_stem, 'weight', None)
        if weight is not None and hasattr(weight, 'shape'):
            out_chans = int(weight.shape[0])
        else:
            # 回退常见值（EffNetV2-S stem 输出常为 32）
            out_chans = 32

    # 用3D stem替换，仅改变第一层，其余网络保持不变
    model.conv_stem = Conv3DStemTo2D(
        in_chans=in_chans,
        out_chans=out_chans,
        kernel_size=(kd, 3, 3),
        stride=(1, 2, 2),
        padding=(1, 1, 1),
        bias=False,
        reduce=reduce,
    )
    return model

class ModelWithSigmoid(nn.Module):
    """
    包装基础模型，在前向输出处加 Sigmoid，用于多标签场景。
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        return torch.sigmoid(logits)

def create_tf_efficientnetv2_s_3d_sigmoid(
    num_classes: int = 14,
    in_chans: int = 1,
    pretrained: bool = False,
    kd: int = 3,
    reduce: Literal['mean', 'max'] = 'mean',
    stem_in_chans_2d: int = 32,
) -> nn.Module:
    """
    创建带 Sigmoid 输出的 3D 版本 EfficientNetV2-S。
    """
    base = create_tf_efficientnetv2_s_3d(
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained=pretrained,
        kd=kd,
        reduce=reduce,
        stem_in_chans_2d=stem_in_chans_2d,
    )
    return ModelWithSigmoid(base)

class MyModel(nn.Module):
    """
    2.5D 封装 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
    - 输入: 4D (B, kd, 384, 384)，通道数需等于 kd
    - 输出: logits (B, num_classes)
    - 说明: timm 会对位置编码插值以支持 384 分辨率
    """
    def __init__(
        self,
        num_classes: int = 14,
        kd: int = 3,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.kd = kd
        # 主干 in_chans 设为 kd，用于接收 2.5D 堆叠的通道
        self.backbone = timm.create_model(
            'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
            num_classes=num_classes,
            pretrained=pretrained,
            in_chans=kd,
            img_size=384,
        )
        if checkpoint_path:
            sd = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
                sd = sd['state_dict']
            # 去除常见前缀，避免键名不一致
            if any(k.startswith('module.') for k in sd.keys()):
                sd = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
            if any(k.startswith('model.') for k in sd.keys()):
                sd = { (k[6:] if k.startswith('model.') else k): v for k, v in sd.items() }
            # 非严格加载以兼容不同 head/分类数
            self.backbone.load_state_dict(sd, strict=False)


    # 输入通道检查由数据管道保证（要求输入 C == kd），模型内部不做裁剪/填充以避免导出 TracerWarning

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
if __name__ == '__main__':
    torch.manual_seed(0)

    # 检查CUDA是否可用；后续在 GPU 上进行前向与导出，并监控显存
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total Memory: {props.total_memory / (1024 ** 3):.2f} GB")
        print(f"Allocated Memory before operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Cached Memory before operation: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    else:
        print("CUDA not available, using CPU.")

    # ===== 仅保留 EVA02 2.5D 导出（4D输入，分辨率 384x384），在 GPU 上运行并监控显存 =====
    num_classes = 14
    kd = 32
    model = MyModel(num_classes=num_classes, kd=kd, pretrained=False).to(device)
    model.eval()

    # 构造示例 4D 输入 (B, C, H, W)，其中 C=kd，H=W=384
    x = torch.randn(2, kd, 384, 384, device=device)

    # 监控: 前向推理前显存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(f"[GPU] Before forward - Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB, Reserved: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")

    # 前向推理
    with torch.no_grad():
        y = model(x)
    print('Output shape:', y.shape)

    # 监控: 前向推理后显存
    if device.type == 'cuda':
        torch.cuda.synchronize()
        print(f"[GPU] After forward  - Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB, Reserved: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB, Peak: {torch.cuda.max_memory_allocated(0) / (1024 ** 3):.2f} GB")

    # 导出 ONNX（输入为 4D，内部仅做通道裁剪/填充，输出 logits）
    onnx_path = 'eva02_base_patch14_2p5d_384_logits.onnx'
    dynamic_axes = {
        'input4d': {0: 'batch'},
        'logits': {0: 'batch'}
    }

    # 监控: 导出前显存
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(f"[GPU] Before export - Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB, Reserved: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")

    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=['input4d'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    # 监控: 导出后显存
    if device.type == 'cuda':
        torch.cuda.synchronize()
        print(f"[GPU] After export  - Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB, Reserved: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB, Peak: {torch.cuda.max_memory_allocated(0) / (1024 ** 3):.2f} GB")

    print(f"Saved ONNX to: {onnx_path}")