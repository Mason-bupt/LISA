from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers import CLIPModel, CLIPTokenizer
except ImportError:
    CLIPModel = None
    CLIPTokenizer = None


@dataclass
class SDMConfig:
    clip_model_name: str = "openai/clip-vit-base-patch32"
    num_gaze_irrelevant_prompts: int = 8
    feature_separation_weight: float = 15.0
    embedding_dim: int = 512


class SDMModule(nn.Module):
    
    def __init__(self, cfg: SDMConfig, input_feature_dim: int):
        super().__init__()
        self.cfg = cfg
        
        if CLIPModel is None or CLIPTokenizer is None:
            raise ImportError(
                "transformers library required: pip install transformers"
            )
        
        self.clip_model = CLIPModel.from_pretrained(cfg.clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        
        self.gaze_irrelevant_templates = [
            "a person with {attribute}",
            "a face with {attribute}",
            "a person wearing {attribute}",
            "a face showing {attribute}",
        ]
        
        self.gaze_irrelevant_attributes = [
            "long hair", "short hair", "curly hair", "straight hair",
            "beard", "mustache", "glasses", "sunglasses",
            "hat", "cap", "makeup", "no makeup",
            "smiling", "frowning", "neutral expression",
            "bright lighting", "dim lighting", "natural lighting",
            "indoor", "outdoor", "blurry", "sharp",
        ]
        
        self._build_prompts()
        
        self.feature_proj = nn.Sequential(
            nn.Linear(input_feature_dim, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
            nn.GELU(),
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
        )
    
    def _build_prompts(self):
        self.prompts = []
        for template in self.gaze_irrelevant_templates:
            for attr in self.gaze_irrelevant_attributes:
                prompt = template.format(attribute=attr)
                self.prompts.append(prompt)
        
        if len(self.prompts) > self.cfg.num_gaze_irrelevant_prompts:
            random.seed(42)
            self.prompts = random.sample(
                self.prompts, self.cfg.num_gaze_irrelevant_prompts
            )
    
    @torch.no_grad()
    def encode_text_prompts(self, device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(
            self.prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=77,
        ).to(device)
        
        text_outputs = self.clip_model.get_text_features(**inputs)
        
        norm = torch.norm(text_outputs, p=2, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        text_features = text_outputs / norm
        
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            text_features = torch.zeros_like(text_features)
            text_features[:, 0] = 1.0
        
        return text_features
    
    def forward(self, gaze_features: torch.Tensor) -> dict[str, torch.Tensor]:
        device = gaze_features.device
        
        gaze_features_proj = self.feature_proj(gaze_features)
        
        norm = torch.norm(gaze_features_proj, p=2, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        gaze_features_proj = gaze_features_proj / norm
        
        if torch.isnan(gaze_features_proj).any() or torch.isinf(gaze_features_proj).any():
            gaze_features_proj = torch.zeros_like(gaze_features_proj)
            gaze_features_proj[:, 0] = 1.0
        
        text_features = self.encode_text_prompts(device)
        
        return {
            "gaze_features_proj": gaze_features_proj,
            "text_features": text_features,
        }


def feature_separation_loss(
    gaze_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if torch.isnan(gaze_features).any() or torch.isinf(gaze_features).any():
        return torch.tensor(0.0, device=gaze_features.device, requires_grad=True)
    if torch.isnan(text_features).any() or torch.isinf(text_features).any():
        return torch.tensor(0.0, device=gaze_features.device, requires_grad=True)
    
    similarity = torch.matmul(gaze_features, text_features.t())
    similarity = torch.clamp(similarity, min=-1.0, max=1.0)
    loss = torch.mean(torch.abs(similarity))
    
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=gaze_features.device, requires_grad=True)
    
    loss = torch.clamp(loss, min=0.0, max=10.0)
    
    return loss
