from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import timm
import torch
from torch import nn


@dataclass 
class ResNet18FeatureConfig:
    """ResNet18 特征输出配置。"""

    model_name: str = "resnet18"
    out_indices: Tuple[int, int] = (1, 3)  # layer2 (索引1) + layer4 (索引3)
    pretrained: bool = True
    checkpoint_path: Optional[str] = None


class ResNet18FeatureExtractor(nn.Module):
    """
    对 timm ResNet18 的轻量封装。

    返回:
        detail (layer2 特征), deep (layer4 特征)
    """

    def __init__(self, cfg: ResNet18FeatureConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ResNet18FeatureConfig()
        self.cfg = cfg
        
        # 使用 timm 创建 ResNet18 模型，输出指定层的特征
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=cfg.out_indices,
        )
        
        self._maybe_load_checkpoint(cfg.checkpoint_path)
        self.detail_idx, self.deep_idx = cfg.out_indices
        
        # 获取特征通道数
        channels = self.backbone.feature_info.channels()
        if len(channels) < 2:
            raise RuntimeError("ResNet18FeatureExtractor 需要至少两个输出特征供 layer2/layer4 使用。")
            
        # feature_info.channels 顺序与 out_indices 一致
        self.detail_channels = channels[0]  # layer2: 128 channels
        self.deep_channels = channels[-1]   # layer4: 512 channels

    def _maybe_load_checkpoint(self, path: Optional[str]) -> None:
        if not path:
            return
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"指定的 ResNet18 checkpoint 不存在: {ckpt_path}")
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features: Sequence[torch.Tensor] = self.backbone(x)
        detail = features[0]  # layer2 特征
        deep = features[1]    # layer4 特征
        return detail, deep


