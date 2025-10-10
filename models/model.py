import math
import torch
from torch import nn
import timm
from typing import List, Union

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        sample = self.act(sample)

        sample = self.linear_2(sample)
        sample = self.act(sample)
        return sample


class AgeSinCosEncoder(nn.Module):
    """
    年龄特征编码器（diffusion风格时间步嵌入 + 线性细化）：
    - 将输入年龄视作“时间步”，用 Timesteps + TimestepEmbedding 生成高维向量
    - 兼容 (B,) 或标量年龄输入
    """
    def __init__(self, embed_dim: int = 128, mean: float = 60.0, std: float = 20.0, max_period: float = 10000.0):
        super().__init__()
        assert embed_dim > 0, "embed_dim 必须为正数"
        self.embed_dim = embed_dim
        # diffusion-style sinusoidal embedding
        self.time_proj = Timesteps(num_channels=embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.time_embedding = TimestepEmbedding(in_channels=embed_dim, time_embed_dim=embed_dim)
        # 可选线性细化
        self.refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """
        age: (B,) 或标量的张量/数值，单位：岁
        return: (B, embed_dim) 年龄嵌入
        """
        # 保证 age 为 (B,) 浮点 1D 张量
        if not isinstance(age, torch.Tensor):
            age = torch.tensor(age, dtype=torch.float32)
        age = age.squeeze()
        if age.dim() == 0:
            age = age[None]
        age = age.to(dtype=torch.float32)

        # diffusion 风格时间步嵌入
        t_emb = self.time_proj(age)              # (B, embed_dim)
        emb = self.time_embedding(t_emb)         # (B, embed_dim)
        return self.refine(emb)


class MyModel(nn.Module):
    """
    ConvNeXt-Large DINOv3 适配 (32,384,384) 输入，并在每个阶段将 sin/cos 年龄嵌入“加到特征上”（Additive Conditioning）。
    - 年龄 -> sin/cos 高维向量（类似 diffusion 的时间步嵌入）
    - 在 ConvNeXt 的 4 个阶段输出后，线性映射到对应通道维，并以加法方式融合：
      y = x + alpha * proj(age_emb)
    """
    def __init__(
        self,
        num_classes: int,
        in_chans: int = 32,
        pretrained: bool = False,
        age_embed_dim: int = 128,
        film_alpha: float = 0.1,
    ):
        super().__init__()
        self.film_alpha = film_alpha

        # 2D 主干：ConvNeXt-Large DINOv3（可设 in_chans=32）
        self.backbone = timm.create_model(
            'convnext_large.dinov3_lvd1689m',
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )

        # 读取 2D 阶段通道数（4 个阶段）
        assert hasattr(self.backbone, "feature_info"), "ConvNeXt 主干缺少 feature_info。"
        self.stage_channels: List[int] = [fi["num_chs"] for fi in self.backbone.feature_info]
        assert len(self.stage_channels) == 4, f"期望 4 个阶段，得到 {len(self.stage_channels)}。"

        # 年龄编码器 + 各阶段 Additive 映射
        self.age_encoder = AgeSinCosEncoder(embed_dim=age_embed_dim)
        self.add_mlps = nn.ModuleList([nn.Linear(age_embed_dim, c) for c in self.stage_channels])

    def forward(self, x: torch.Tensor, age: Union[torch.Tensor, int, float]) -> torch.Tensor:
        """
        输入：
        - x: (B, 32, 384, 384) 多切片 2D 体图
        - age: (B,) 张量或 Python 标量（int/float），单位：岁

        输出：
        - logits: (B, num_classes)
        """
        # 检查输入通道与主干一致
        in_chs_expected = self.backbone.stem[0].in_channels
        assert x.dim() == 4 and x.shape[1] == in_chs_expected, f"期望输入形状为 (B, {in_chs_expected}, H, W)，得到 {tuple(x.shape)}"

        # 处理年龄输入为 (B,) float tensor
        if not isinstance(age, torch.Tensor):
            age = torch.tensor(age, dtype=x.dtype, device=x.device).repeat(x.shape[0])
        else:
            age = age.to(dtype=x.dtype, device=x.device)
            if age.dim() == 0:
                age = age.repeat(x.shape[0])

        # 年龄嵌入
        age_feat = self.age_encoder(age)  # (B, age_embed_dim)

        # 2D 干部分阶段前向 + Additive 融合
        x2d = self.backbone.stem(x)
        for i, stage in enumerate(self.backbone.stages):
            x2d = stage(x2d)
            add_vec = self.add_mlps[i](age_feat)  # (B, C_i)
            x2d = x2d + self.film_alpha * add_vec.unsqueeze(-1).unsqueeze(-1)

        # 末端归一化 + 分类头
        x2d = self.backbone.norm_pre(x2d)
        logits = self.backbone.forward_head(x2d)
        return logits


def print_gpu_info(prefix: str = "") -> None:
    """
    简单输出 GPU 监控信息：
    - 设备索引与名称
    - 当前已分配显存（allocated）
    - 当前已保留显存（reserved）
    - 设备总显存（total）
    当 CUDA 不可用时打印提示。
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        name = torch.cuda.get_device_name(dev)
        mem_alloc = torch.cuda.memory_allocated(dev) / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
        mem_total = props.total_memory / (1024 ** 2)
        print(f"[GPU] {prefix} | idx={dev} name={name} | alloc={mem_alloc:.1f}MB reserved={mem_reserved:.1f}MB total={mem_total:.0f}MB")
    else:
        print("[GPU] CUDA not available")
        
if __name__ == '__main__':
    # 简单自测 + GPU 监控
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_info("startup")

    model = MyModel(num_classes=14, in_chans=32, pretrained=False).to(device)
    print_gpu_info("after model.to(device)")

    x = torch.randn(1, 32, 384, 384, device=device)
    age = torch.tensor([55.0], device=device)
    print_gpu_info("after inputs allocated")

    with torch.no_grad():
        y = model(x, age)

    print("logits shape:", y.shape)
    print_gpu_info("after forward")
