from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import timm
import torch
from torch import nn


@dataclass 
class ResNet18FeatureConfig:
    model_name: str = "resnet18"
    out_indices: Tuple[int, int] = (1, 3)
    pretrained: bool = True
    checkpoint_path: Optional[str] = None


class ResNet18FeatureExtractor(nn.Module):

    def __init__(self, cfg: ResNet18FeatureConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ResNet18FeatureConfig()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=cfg.out_indices,
        )
        
        self._maybe_load_checkpoint(cfg.checkpoint_path)
        self.detail_idx, self.deep_idx = cfg.out_indices
        
        channels = self.backbone.feature_info.channels()
        if len(channels) < 2:
            raise RuntimeError("ResNet18FeatureExtractor requires at least two output features for layer2/layer4.")
            
        self.detail_channels = channels[0]
        self.deep_channels = channels[-1]

    def _maybe_load_checkpoint(self, path: Optional[str]) -> None:
        if not path:
            return
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ResNet18 checkpoint not found: {ckpt_path}")
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features: Sequence[torch.Tensor] = self.backbone(x)
        detail = features[0]
        deep = features[1]
        return detail, deep
