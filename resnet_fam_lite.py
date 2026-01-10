from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .resnet18_backbone import ResNet18FeatureConfig, ResNet18FeatureExtractor
from .clip_gaze import CLIPGazeModule, CLIPGazeConfig


class FeatureAlign(nn.Module):
    """
    对齐 layer2/layer4 特征。
    优化点: 
      1. 减少 permute 次数，尽量保持内存连续性。
      2. 使用 Bias=False 配合 BatchNorm (减少显存占用)。
    """

    def __init__(self, target_channels: int, guide_channels: int, fused_channels: int) -> None:
        super().__init__()
        self.target_channels = target_channels
        self.guide_channels = guide_channels
        
        # 预先定义好投影层
        self.target_proj = nn.Sequential(
            nn.Conv2d(target_channels, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.GELU(),
        )
        self.guide_proj = nn.Sequential(
            nn.Conv2d(guide_channels, fused_channels, 1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.GELU(),
        )

    def forward(self, target: torch.Tensor, guide: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 快速通道检查与转换，减少 getattr 开销
        target = self._ensure_chw(target, self.target_channels)
        guide = self._ensure_chw(guide, self.guide_channels)

        # 仅在尺寸不匹配时插值 (通常 layer4 尺寸小于 layer2)
        if target.shape[-2:] != guide.shape[-2:]:
            guide = F.interpolate(guide, size=target.shape[-2:], mode="bilinear", align_corners=False)

        return self.target_proj(target), self.guide_proj(guide)

    @staticmethod
    def _ensure_chw(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
        """优化后的维度检查"""
        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
        
        # ResNet输出已经是BCHW格式
        if tensor.shape[1] == expected_channels:
            return tensor
            
        raise ValueError(f"Channel mismatch. Expected {expected_channels}, got shape {tensor.shape}")


class FrequencyModulation(nn.Module):
    """
    利用频域替换低频信息。
    修改点: 
     1. 引入可学习 alpha（sigmoid 约束到 (0,1)）。
     2. 保留 Mask 缓存 (避免频繁 alloc)。
     3. 修复低频 Mask 逻辑 (同时保留正负频率，消除伪影)。
     4. 确保 Mask dtype 为 complex 以兼容 torch.lerp / 复数运算。
    """

    def __init__(self, low_freq_ratio: float = 0.25, init_alpha: float = 0.3) -> None:
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        # 注册 buffer 用于缓存，persistent=False 表示不保存到模型权重文件中
        self.register_buffer("cached_mask", None, persistent=False)
        # 可学习的 alpha（用 sigmoid 约束到 (0,1)）
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, target: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        B, C, H, W = target.shape
        
        # 1. 使用 rfft2 (实数到复数)，利用共轭对称性
        fft_target = torch.fft.rfft2(target, dim=(-2, -1)) 
        fft_guide = torch.fft.rfft2(guide, dim=(-2, -1))
        
        # 2. 获取 Mask (O(1) 缓存读取)
        mask = self._get_mask(H, W, target.device, fft_target.dtype)
        
        # 3. alpha 控制融合强度（sigmoid -> (0,1)）
        alpha_val = torch.sigmoid(self.alpha).to(device=target.device, dtype=fft_target.real.dtype)
        
        # 4. 频域融合（使用可学习 alpha）
        # 注意：mask 是复数 dtype，alpha_val 是实数，二者可乘
        fused_fft = fft_target * (1.0 - alpha_val * mask) + fft_guide * (alpha_val * mask)
        
        # 5. 逆变换回实数域
        fused = torch.fft.irfft2(fused_fft, s=(H, W), dim=(-2, -1))
        return fused

    def _get_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """生成并缓存 Mask，确保 dtype 匹配"""
        # RFFT 的宽度是 w // 2 + 1
        w_rfft = w // 2 + 1
        
        # 检查缓存: 尺寸、设备、类型都要匹配
        if (self.cached_mask is not None and 
            self.cached_mask.shape[-2:] == (h, w_rfft) and 
            self.cached_mask.device == device and
            self.cached_mask.dtype == dtype):
            return self.cached_mask

        h_cut = max(1, int(h * self.low_freq_ratio))
        w_cut = max(1, int(w * self.low_freq_ratio))

        # 创建 Mask，直接使用传入的 complex dtype
        mask = torch.zeros((1, 1, h, w_rfft), device=device, dtype=dtype)
        
        # 填充低频区域 (赋值 1.0 会自动转换为 1.0+0j)
        # 左上角 (正频率)
        mask[:, :, :h_cut, :w_cut] = 1.0 
        # 左下角 (负频率) - 真正的低通滤波，防止 Ringing Artifacts
        mask[:, :, -h_cut:, :w_cut] = 1.0 

        self.cached_mask = mask
        return mask


class AttentionModulation(nn.Module):
    """
    语义空间注意力。
    修改点: 将 residual 默认值增加到 0.5，减轻对主特征的抑制（防止过度衰减）。
    """

    def __init__(self, channels: int, reduction: int = 4, residual: float = 0.5) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.residual = residual
        self.mask_net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, fm: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        mask = self.mask_net(guide)
        # 避免 inplace 操作破坏梯度链；使用加法得到 scale
        scale = mask + self.residual
        return fm * scale 


class FAMFusion(nn.Module):
    def __init__(self, channels: int, low_freq_ratio: float = 0.25) -> None:
        super().__init__()
        self.freq = FrequencyModulation(low_freq_ratio=low_freq_ratio)
        self.attn = AttentionModulation(channels)


class RegressionHead(nn.Module):
    """
    CoordConv + 卷积残差 + 保留空间信息 -> GAP/MLP。
    修改点:
      1. 在 GAP 前加入坐标通道（CoordConv 思路）以保留空间偏置信息。
      2. 将 pooling 保留为小空间 (2x2) 而不是 1x1，以保留空间提示。
    """
    def __init__(self, channels: int, hidden_dim: int = 512, output_dim: int = 2) -> None:
        super().__init__()
        self.channels = channels
        # 主体卷积块
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # 将 concat 后的 channels(原C + 2coord) 投影回 channels
        self.coord_proj = nn.Conv2d(channels + 2, channels, 1, bias=False)
        self.coord_bn = nn.BatchNorm2d(channels)
        # MLP 最终分类器（输入为 channels * 2 * 2）
        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _build_coords(batch: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        生成 coord maps，范围 [-1,1]
        返回 shape: [B, 2, H, W]
        """
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        
        # 使用 meshgrid 生成网格 (indexing='ij' 对应 y, x 顺序)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        
        # 堆叠得到 [2, H, W]，其中 channel 0 是 x，channel 1 是 y
        coords = torch.stack([xx, yy], dim=0)
        
        # 增加 batch 维度并扩展: [1, 2, H, W] -> [B, 2, H, W]
        coords = coords.unsqueeze(0).expand(batch, -1, -1, -1)
        
        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.conv_block(x)  # [B, C, H, W]

        # 生成 coord 并 concat
        coords = self._build_coords(B, H, W, x.device, x.dtype)
        x = torch.cat([x, coords], dim=1)  # [B, C+2, H, W]

        # 投影回 channels 并 BN + act
        x = self.coord_proj(x)
        x = self.coord_bn(x)
        x = F.gelu(x)

        # 保留少量空间信息 (2x2)，然后 flatten
        # 相比 GAP (1x1)，这里保留了上下左右的空间分布特征
        x = F.adaptive_avg_pool2d(x, (2, 2)).flatten(1)  # [B, C*4]

        return self.mlp(x)


@dataclass
class ResNetFAMLiteConfig:
    """ResNet18-FAM-Lite 配置"""
    backbone: ResNet18FeatureConfig = field(default_factory=ResNet18FeatureConfig)
    
    # 针对 ResNet18 推荐 256，避免瓶颈
    fusion_channels: int = 256       
    low_freq_ratio: float = 0.25
    # 针对 ResNet18 推荐 256 或 512
    head_hidden_dim: int = 256       
    
    use_clip_gaze: bool = True
    clip_gaze_config: Optional[CLIPGazeConfig] = None


class ResNetFAMLite(nn.Module):
    """
    ResNet18-FAM-Lite 模型
    """
    def __init__(self, cfg: Optional[ResNetFAMLiteConfig] = None) -> None:
        super().__init__()
        cfg = cfg or ResNetFAMLiteConfig()
        self.cfg = cfg
        
        self.backbone = ResNet18FeatureExtractor(cfg.backbone)
        
        self.align = FeatureAlign(
            target_channels=self.backbone.detail_channels,
            guide_channels=self.backbone.deep_channels,
            fused_channels=cfg.fusion_channels,
        )
        
        # 使用修改后的 FAMFusion 结构（其中 Frequency 为可学习 alpha）
        self.fusion = FAMFusion(cfg.fusion_channels, low_freq_ratio=cfg.low_freq_ratio)
        
        # 使用 CoordConv + 保留空间信息的 RegressionHead
        self.head = RegressionHead(cfg.fusion_channels, hidden_dim=cfg.head_hidden_dim)
        
        self.use_clip_gaze = cfg.use_clip_gaze
        if self.use_clip_gaze:
            clip_cfg = cfg.clip_gaze_config or CLIPGazeConfig()
            self.clip_gaze = CLIPGazeModule(clip_cfg, input_feature_dim=cfg.fusion_channels)
        else:
            self.clip_gaze = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Backbone Inference
        detail, deep = self.backbone(x)
        
        # 2. Alignment & Fusion (RFFT + learnable alpha 加速)
        target_feat, guide_feat = self.align(detail, deep)
        
        # Frequency fusion (使用 FrequencyModulation 的 forward)
        fused_freq = self.fusion.freq(target_feat, guide_feat) # [B, C, H, W]
        # Attention modulation
        fused = self.fusion.attn(fused_freq, guide_feat) # [B, C, H, W]
        
        # 3. Head Inference (包含 Conv -> CoordConv 投影 -> pool -> MLP)
        gaze = self.head(fused)

        output = {"gaze": gaze, "fused_feature": fused}
        
        # 4. CLIP-Gaze (如果启用)
        if self.use_clip_gaze and self.clip_gaze is not None:
            # 优化：做一个简单的 GAP 用于 CLIP 投影
            gaze_feat = F.adaptive_avg_pool2d(fused, 1).flatten(1)
            
            clip_output = self.clip_gaze(gaze_feat)
            output["gaze_features_proj"] = clip_output["gaze_features_proj"]
            output["text_features"] = clip_output["text_features"]
        
        return output


