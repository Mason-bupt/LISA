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
class CLIPGazeConfig:
    """CLIP-Gaze配置 - 最小化版本，只保留核心功能"""
    clip_model_name: str = "openai/clip-vit-base-patch32"
    num_gaze_irrelevant_prompts: int = 8
    feature_separation_weight: float = 15.0  # 已调整为与回归Loss（度数单位）同数量级
    embedding_dim: int = 512


class CLIPGazeModule(nn.Module):
    """
    CLIP-Gaze核心模块 - 简化版本
    使用CLIP文本编码器生成与注视无关的特征，并通过特征分离损失push away这些特征
    """
    
    def __init__(self, cfg: CLIPGazeConfig, input_feature_dim: int):
        super().__init__()
        self.cfg = cfg
        
        if CLIPModel is None or CLIPTokenizer is None:
            raise ImportError(
                "需要安装transformers库: pip install transformers"
            )
        
        # 加载CLIP模型和分词器
        self.clip_model = CLIPModel.from_pretrained(cfg.clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
        
        # 冻结CLIP模型参数（只使用其预训练知识）
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        
        # 定义与注视无关的文本提示模板
        self.gaze_irrelevant_templates = [
            "a person with {attribute}",
            "a face with {attribute}",
            "a person wearing {attribute}",
            "a face showing {attribute}",
        ]
        
        # 与注视无关的属性列表
        self.gaze_irrelevant_attributes = [
            "long hair", "short hair", "curly hair", "straight hair",
            "beard", "mustache", "glasses", "sunglasses",
            "hat", "cap", "makeup", "no makeup",
            "smiling", "frowning", "neutral expression",
            "bright lighting", "dim lighting", "natural lighting",
            "indoor", "outdoor", "blurry", "sharp",
        ]
        
        # 生成完整的提示列表
        self._build_prompts()
        
        # 特征投影层：将gaze特征投影到CLIP特征空间
        self.feature_proj = nn.Sequential(
            nn.Linear(input_feature_dim, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
            nn.GELU(),
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
        )
    
    def _build_prompts(self):
        """构建与注视无关的文本提示"""
        self.prompts = []
        for template in self.gaze_irrelevant_templates:
            for attr in self.gaze_irrelevant_attributes:
                prompt = template.format(attribute=attr)
                self.prompts.append(prompt)
        
        # 限制提示数量
        if len(self.prompts) > self.cfg.num_gaze_irrelevant_prompts:
            random.seed(42)
            self.prompts = random.sample(
                self.prompts, self.cfg.num_gaze_irrelevant_prompts
            )
    
    @torch.no_grad()
    def encode_text_prompts(self, device: torch.device) -> torch.Tensor:
        """
        编码文本提示为特征向量
        
        Returns:
            text_features: [num_prompts, embedding_dim] 与注视无关的文本特征
        """
        # 分词
        inputs = self.tokenizer(
            self.prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=77,  # CLIP的最大序列长度
        ).to(device)
        
        # 获取文本嵌入
        text_outputs = self.clip_model.get_text_features(**inputs)
        
        # 安全的归一化：处理零向量或接近零向量的情况
        norm = torch.norm(text_outputs, p=2, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        text_features = text_outputs / norm
        
        # 检查NaN和Inf
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            # 如果出现NaN/Inf，使用单位向量
            text_features = torch.zeros_like(text_features)
            text_features[:, 0] = 1.0
        
        return text_features
    
    def forward(self, gaze_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            gaze_features: [batch_size, feature_dim] 注视相关特征
        
        Returns:
            dict包含:
                - gaze_features_proj: 投影后的注视特征
                - text_features: 与注视无关的文本特征
        """
        device = gaze_features.device
        
        # 投影gaze特征到CLIP特征空间
        gaze_features_proj = self.feature_proj(gaze_features)
        
        # 安全的归一化：处理零向量或接近零向量的情况
        norm = torch.norm(gaze_features_proj, p=2, dim=1, keepdim=True)
        # 如果norm太小（接近0），使用单位向量避免除零
        norm = torch.clamp(norm, min=1e-8)
        gaze_features_proj = gaze_features_proj / norm
        
        # 检查NaN和Inf
        if torch.isnan(gaze_features_proj).any() or torch.isinf(gaze_features_proj).any():
            # 如果出现NaN/Inf，使用零向量并发出警告
            gaze_features_proj = torch.zeros_like(gaze_features_proj)
            gaze_features_proj[:, 0] = 1.0  # 设置为单位向量
        
        # 编码文本提示
        text_features = self.encode_text_prompts(device)  # [num_prompts, embedding_dim]
        
        return {
            "gaze_features_proj": gaze_features_proj,
            "text_features": text_features,
        }


def feature_separation_loss(
    gaze_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    特征分离损失：最小化注视相关特征与注视无关特征之间的相似度
    
    这是CLIP-Gaze的核心思想：让注视特征与外观特征正交（orthogonal），即互不相关
    
    改进：使用绝对值损失 |similarity|，对小的相似度变化更敏感
    - 当 similarity = 0 时，loss = 0（完美分离）
    - 当 similarity = ±0.01 时，loss = 0.01（有梯度！比平方损失敏感100倍）
    - 当 similarity = ±1 时，loss = 1（完全相关，需要惩罚）
    
    相比 similarity² 的优势：
    - 对接近0的相似度变化更敏感，梯度更明显
    - 有助于优化器在训练初期更新随机初始化的投影层
    
    Args:
        gaze_features: [batch_size, embedding_dim] 注视相关特征（已归一化）
        text_features: [num_prompts, embedding_dim] 与注视无关的文本特征（已归一化）
        temperature: 温度参数（保留以保持接口兼容，但不再使用）
    
    Returns:
        loss: 标量损失值（范围约0-1，与回归损失的度数单位同量级）
    """
    # 检查输入是否包含NaN或Inf
    if torch.isnan(gaze_features).any() or torch.isinf(gaze_features).any():
        return torch.tensor(0.0, device=gaze_features.device, requires_grad=True)
    if torch.isnan(text_features).any() or torch.isinf(text_features).any():
        return torch.tensor(0.0, device=gaze_features.device, requires_grad=True)
    
    # 计算相似度矩阵 [batch_size, num_prompts]
    # 归一化向量的点积范围是[-1, 1]
    similarity = torch.matmul(gaze_features, text_features.t())
    
    # Clamp相似度到有效范围，避免数值问题
    similarity = torch.clamp(similarity, min=-1.0, max=1.0)
    
    # 使用绝对值损失：对小的相似度变化更敏感
    # 相比 similarity²，当 similarity = 0.01 时：
    #   - similarity² = 0.0001（梯度很小）
    #   - |similarity| = 0.01（梯度明显）
    loss = torch.mean(torch.abs(similarity))
    
    # 检查损失是否为NaN或Inf
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=gaze_features.device, requires_grad=True)
    
    # Clamp损失值到合理范围，避免极端值
    loss = torch.clamp(loss, min=0.0, max=10.0)
    
    return loss
