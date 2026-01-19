from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .resnet18_backbone import ResNet18FeatureConfig, ResNet18FeatureExtractor
from .sdm import SDMModule, SDMConfig


class FeatureAlign(nn.Module):

    def __init__(self, target_channels: int, guide_channels: int, fused_channels: int) -> None:
        super().__init__()
        self.target_channels = target_channels
        self.guide_channels = guide_channels
        
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
        target = self._ensure_chw(target, self.target_channels)
        guide = self._ensure_chw(guide, self.guide_channels)
        if target.shape[-2:] != guide.shape[-2:]:
            guide = F.interpolate(guide, size=target.shape[-2:], mode="bilinear", align_corners=False)

        return self.target_proj(target), self.guide_proj(guide)

    @staticmethod
    def _ensure_chw(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
        
        if tensor.shape[1] == expected_channels:
            return tensor
            
        raise ValueError(f"Channel mismatch. Expected {expected_channels}, got shape {tensor.shape}")


class FrequencyModulation(nn.Module):

    def __init__(self, low_freq_ratio: float = 0.25, init_alpha: float = 0.3) -> None:
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        self.register_buffer("cached_mask", None, persistent=False)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, target: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        B, C, H, W = target.shape
        
        fft_target = torch.fft.rfft2(target, dim=(-2, -1)) 
        fft_guide = torch.fft.rfft2(guide, dim=(-2, -1))
        
        mask = self._get_mask(H, W, target.device, fft_target.dtype)
        
        alpha_val = torch.sigmoid(self.alpha).to(device=target.device, dtype=fft_target.real.dtype)
        
        fused_fft = fft_target * (1.0 - alpha_val * mask) + fft_guide * (alpha_val * mask)
        
        fused = torch.fft.irfft2(fused_fft, s=(H, W), dim=(-2, -1))
        return fused

    def _get_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        w_rfft = w // 2 + 1
        if (self.cached_mask is not None and 
            self.cached_mask.shape[-2:] == (h, w_rfft) and 
            self.cached_mask.device == device and
            self.cached_mask.dtype == dtype):
            return self.cached_mask

        h_cut = max(1, int(h * self.low_freq_ratio))
        w_cut = max(1, int(w * self.low_freq_ratio))

        mask = torch.zeros((1, 1, h, w_rfft), device=device, dtype=dtype)
        mask[:, :, :h_cut, :w_cut] = 1.0 
        mask[:, :, -h_cut:, :w_cut] = 1.0 

        self.cached_mask = mask
        return mask


class AttentionModulation(nn.Module):

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
        scale = mask + self.residual
        return fm * scale 


class FAMFusion(nn.Module):
    def __init__(self, channels: int, low_freq_ratio: float = 0.25) -> None:
        super().__init__()
        self.freq = FrequencyModulation(low_freq_ratio=low_freq_ratio)
        self.attn = AttentionModulation(channels)


class RegressionHead(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 512, output_dim: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.coord_proj = nn.Conv2d(channels + 2, channels, 1, bias=False)
        self.coord_bn = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _build_coords(batch: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=0)
        coords = coords.unsqueeze(0).expand(batch, -1, -1, -1)
        
        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.conv_block(x)

        coords = self._build_coords(B, H, W, x.device, x.dtype)
        x = torch.cat([x, coords], dim=1)

        x = self.coord_proj(x)
        x = self.coord_bn(x)
        x = F.gelu(x)

        x = F.adaptive_avg_pool2d(x, (2, 2)).flatten(1)

        return self.mlp(x)


@dataclass
class ResNetFAMLiteConfig:
    backbone: ResNet18FeatureConfig = field(default_factory=ResNet18FeatureConfig)
    
    fusion_channels: int = 256       
    low_freq_ratio: float = 0.25
    head_hidden_dim: int = 256       
    
    use_sdm: bool = True
    sdm_config: Optional[SDMConfig] = None


class ResNetFAMLite(nn.Module):
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
        
        self.fusion = FAMFusion(cfg.fusion_channels, low_freq_ratio=cfg.low_freq_ratio)
        
        self.head = RegressionHead(cfg.fusion_channels, hidden_dim=cfg.head_hidden_dim)
        
        self.use_sdm = cfg.use_sdm
        if self.use_sdm:
            sdm_cfg = cfg.sdm_config or SDMConfig()
            self.sdm = SDMModule(sdm_cfg, input_feature_dim=cfg.fusion_channels)
        else:
            self.sdm = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        detail, deep = self.backbone(x)
        
        target_feat, guide_feat = self.align(detail, deep)
        fused_freq = self.fusion.freq(target_feat, guide_feat)
        fused = self.fusion.attn(fused_freq, guide_feat)
        
        gaze = self.head(fused)

        output = {"gaze": gaze, "fused_feature": fused}
        
        if self.use_sdm and self.sdm is not None:
            gaze_feat = F.adaptive_avg_pool2d(fused, 1).flatten(1)
            
            sdm_output = self.sdm(gaze_feat)
            output["gaze_features_proj"] = sdm_output["gaze_features_proj"]
            output["text_features"] = sdm_output["text_features"]
        
        return output
