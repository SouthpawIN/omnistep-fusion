"""
OmniStep Fusion — Dimension Projection Layers
Bridges 2560-dim (AceStep) ↔ 4096-dim (Cosmos)

These projections map between the two model's hidden spaces,
allowing encoders/decoders from either model to work with the
fused backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectionConfig:
    """Configuration for dimension projection."""
    input_dim: int = 2560      # AceStep hidden size
    output_dim: int = 4096     # Cosmos hidden size
    num_layers: int = 2        # Depth of projection MLP
    dropout: float = 0.0
    norm_type: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    activation: str = "silu"    # Match both models' activation
    init_scale: float = 0.02   # Xavier init scale


class RMSNorm(nn.Module):
    """RMSNorm matching Qwen3's implementation."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class DimensionProjection(nn.Module):
    """
    Projects hidden states from one model's dimension to another.
    
    Architecture:
        input → Linear → SiLU → [Linear → SiLU] × (num_layers-1) → Norm → output
    
    The projection preserves semantic meaning while mapping between
    representation spaces. Trained on parallel embeddings from both models.
    """
    
    def __init__(self, config: ProjectionConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.input_dim
        
        for i in range(config.num_layers):
            out_dim = config.output_dim if i == config.num_layers - 1 else config.output_dim * 2
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            if i < config.num_layers - 1:
                layers.append(nn.SiLU())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
            in_dim = out_dim
        
        self.proj = nn.Sequential(*layers)
        
        # Norm layer matching Qwen3's style
        if config.norm_type == "rmsnorm":
            self.norm = RMSNorm(config.output_dim)
        else:
            self.norm = nn.LayerNorm(config.output_dim)
        
        # Initialize with small weights for stable training
        self._init_weights(config.init_scale)
    
    def _init_weights(self, scale: float):
        """Xavier-style initialization for stable training start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states of shape (..., input_dim)
        Returns:
            Projected hidden states of shape (..., output_dim)
        """
        return self.norm(self.proj(x))


class BidirectionalProjection(nn.Module):
    """
    Bidirectional projection between two model spaces.
    Useful for encoder bridges where information flows both ways.
    """
    
    def __init__(self, dim_a: int = 4096, dim_b: int = 2560, config: Optional[ProjectionConfig] = None):
        super().__init__()
        
        cfg_a2b = ProjectionConfig(input_dim=dim_a, output_dim=dim_b)
        cfg_b2a = ProjectionConfig(input_dim=dim_b, output_dim=dim_a)
        
        self.a_to_b = DimensionProjection(cfg_a2b)
        self.b_to_a = DimensionProjection(cfg_b2a)
    
    def forward(self, x: torch.Tensor, direction: str = "a_to_b") -> torch.Tensor:
        if direction == "a_to_b":
            return self.a_to_b(x)
        elif direction == "b_to_a":
            return self.b_to_a(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")


class RoPEAdapter(nn.Module):
    """
    Adapts between standard RoPE (AceStep, θ=1M) and M-RoPE (Cosmos, θ=5M).
    
    M-RoPE uses 3 sections [24, 20, 20] for multimodal positions (text, height, width).
    Standard RoPE uses 1 section for all dimensions.
    
    This adapter learns to map standard positions into M-RoPE's section layout.
    """
    
    def __init__(self, head_dim: int = 128, num_sections: int = 3):
        super().__init__()
        self.head_dim = head_dim
        self.num_sections = num_sections
        
        # Section dimensions: [24, 20, 20] for Cosmos M-RoPE
        # These sum to 64, and head_dim=128 uses pairs, so 64 dims get position encoding
        self.section_dims = [24, 20, 20]
        
        # Learnable section weights for mapping standard RoPE → M-RoPE
        self.section_weights = nn.Parameter(torch.ones(num_sections))
        
        # Frequency scaling to bridge θ=1M → θ=5M
        self.freq_scale = nn.Parameter(torch.tensor(5.0))
    
    def forward(self, positions: torch.Tensor, modality: str = "text") -> torch.Tensor:
        """
        Adapt positions for M-RoPE compatibility.
        
        Args:
            positions: Standard position IDs (batch, seq_len)
            modality: "text", "vision", or "audio"
        Returns:
            Adapted positions for M-RoPE
        """
        # Scale frequencies
        scaled_pos = positions.float() * self.freq_scale.abs()
        
        # Map to M-RoPE sections
        if modality == "text":
            # Text: use section 0 for all positions
            adapted = scaled_pos.unsqueeze(-1).expand(*scaled_pos.shape, self.num_sections)
            adapted[..., 1:] = 0  # Only text section active
        elif modality == "vision":
            # Vision: use all 3 sections (height, width, temporal)
            adapted = scaled_pos.unsqueeze(-1).expand(*scaled_pos.shape, self.num_sections)
        elif modality == "audio":
            # Audio: use sections 0 and 1 (time, frequency)
            adapted = scaled_pos.unsqueeze(-1).expand(*scaled_pos.shape, self.num_sections)
            adapted[..., 2] = 0  # No spatial dimension for audio
        else:
            adapted = scaled_pos.unsqueeze(-1).expand(*scaled_pos.shape, self.num_sections)
        
        # Apply section weights
        adapted = adapted * self.section_weights.abs()
        
        return adapted


class AudioEncoderBridge(nn.Module):
    """
    Bridges AceStep's audio encoder output into Cosmos's backbone space.
    
    AceStep Audio Encoder → 2560-dim → [Bridge] → 4096-dim → Cosmos Backbone
    """
    
    def __init__(self):
        super().__init__()
        config = ProjectionConfig(input_dim=2560, output_dim=4096, num_layers=2)
        self.projection = DimensionProjection(config)
        
        # Modality embedding to distinguish audio tokens from text/vision
        self.modality_embed = nn.Parameter(torch.randn(4096) * 0.02)
    
    def forward(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_hidden: AceStep encoder output (batch, seq_len, 2560)
        Returns:
            Projected audio features (batch, seq_len, 4096)
        """
        projected = self.projection(audio_hidden)
        return projected + self.modality_embed


class AudioDecoderBridge(nn.Module):
    """
    Bridges Cosmos backbone output into AceStep DiT's conditioning space.
    
    Cosmos Backbone → 4096-dim → [Bridge] → 1024-dim → AceStep DiT
    """
    
    def __init__(self):
        super().__init__()
        config = ProjectionConfig(input_dim=4096, output_dim=1024, num_layers=2)
        self.projection = DimensionProjection(config)
    
    def forward(self, backbone_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            backbone_hidden: Cosmos backbone output (batch, seq_len, 4096)
        Returns:
            DiT conditioning (batch, seq_len, 1024)
        """
        return self.projection(backbone_hidden)


# === Training Utilities ===

class ProjectionTrainer:
    """
    Trains projection layers using parallel embeddings from both models.
    
    Strategy:
    1. Generate embeddings from both models on same input
    2. Train projection to map one to the other
    3. Loss = cosine_sim + MSE + task_loss
    """
    
    @staticmethod
    def compute_loss(
        projected: torch.Tensor,
        target: torch.Tensor,
        task_loss: Optional[torch.Tensor] = None,
        cosine_weight: float = 0.5,
        mse_weight: float = 0.3,
        task_weight: float = 0.2,
    ) -> torch.Tensor:
        """
        Combined loss for projection training.
        
        Args:
            projected: Projection output (batch, seq_len, dim)
            target: Target embeddings (batch, seq_len, dim)
            task_loss: Optional downstream task loss
            weights: Loss component weights
        """
        # Cosine similarity loss (maximize)
        cos_sim = F.cosine_similarity(projected, target, dim=-1).mean()
        cos_loss = 1.0 - cos_sim
        
        # MSE loss
        mse_loss = F.mse_loss(projected, target)
        
        total = cosine_weight * cos_loss + mse_weight * mse_loss
        
        if task_loss is not None:
            total = total + task_weight * task_loss
        
        return total
