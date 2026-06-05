"""
OmniStep Fusion — Unified Model Class
The fused omni-modal model: see, hear, generate video, generate music, think.

This is the final assembled model that combines:
- Cosmos3-Nano backbone (vision, video, text)
- AceStep backbone (music, audio)
- All projection layers
- All encoders and decoders
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .projection import (
    DimensionProjection, ProjectionConfig,
    AudioEncoderBridge, AudioDecoderBridge, RoPEAdapter
)


@dataclass
class OmniStepConfig:
    """Configuration for the fused OmniStep model."""
    # Backbone
    hidden_size: int = 4096
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    
    # Vocabulary
    vocab_size: int = 217204  # Merged: 151K (Cosmos) + 65K (music)
    
    # Encoders
    vision_encoder_dim: int = 1152
    audio_encoder_dim: int = 2560
    
    # Decoders
    video_diffusion_dim: int = 4096
    audio_dit_dim: int = 1024  # AceStep DiT conditioning dim
    
    # RoPE
    rope_theta: float = 5000000
    max_position_embeddings: int = 262144


class OmniStepModel(nn.Module):
    """
    The fused OmniStep model — one model to rule them all.
    
    Modalities:
        Input:  Text, Vision (images/video), Audio (music/speech)
        Output: Text, Video (diffusion), Audio (diffusion)
    
    Architecture:
        Shared Qwen3 backbone (36 layers, hidden=4096)
        + Vision encoder bridge (Cosmos ViT → backbone)
        + Audio encoder bridge (AceStep encoder → backbone)
        + Text head (backbone → tokens)
        + Video diffusion expert (backbone → Cosmos diffusion)
        + Audio diffusion bridge (backbone → AceStep DiT)
    """
    
    def __init__(self, config: OmniStepConfig):
        super().__init__()
        self.config = config
        
        # === Embeddings ===
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # === Projection Bridges ===
        self.audio_encoder_bridge = AudioEncoderBridge()
        self.audio_decoder_bridge = AudioDecoderBridge()
        
        # Vision projection (Cosmos already has this, but we include for completeness)
        self.vision_projection = nn.Linear(
            config.vision_encoder_dim, config.hidden_size, bias=False
        )
        
        # === RoPE Adapter ===
        self.rope_adapter = RoPEAdapter(head_dim=config.head_dim)
        
        # === Backbone (placeholder — loaded from fused weights) ===
        # In practice, this is the 36-layer Qwen3 transformer
        # We define the interface here, actual weights come from fusion
        self.backbone = None  # Set during loading
        
        # === Output Heads ===
        # Text output
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Note: Video and Audio diffusion experts are separate models
        # that receive conditioning from the backbone via bridges.
        # They are NOT part of the main forward pass.
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        modality: str = "text",
    ) -> Dict:
        """
        Forward pass through the fused model.
        
        Args:
            input_ids: Text token IDs (batch, seq_len)
            vision_features: Vision encoder output (batch, vision_seq, 1152)
            audio_features: Audio encoder output (batch, audio_seq, 2560)
            attention_mask: Attention mask
            labels: Labels for training
            modality: Current modality ("text", "vision", "audio")
        
        Returns:
            Dict with logits, hidden_states, and modality-specific outputs
        """
        # Get text embeddings
        hidden_states = self.text_embed(input_ids)
        
        # Add vision features if present
        if vision_features is not None:
            vision_proj = self.vision_projection(vision_features)
            # Insert vision features at appropriate positions
            # (In practice, this uses Cosmos's vision token insertion logic)
            hidden_states = self._insert_modality_features(
                hidden_states, vision_proj, "vision"
            )
        
        # Add audio features if present
        if audio_features is not None:
            audio_proj = self.audio_encoder_bridge(audio_features)
            hidden_states = self._insert_modality_features(
                hidden_states, audio_proj, "audio"
            )
        
        # Run through backbone
        if self.backbone is not None:
            backbone_output = self.backbone(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = backbone_output.last_hidden_state
        
        # Get text logits
        logits = self.lm_head(hidden_states)
        
        # Prepare output
        output = {
            'logits': logits,
            'hidden_states': hidden_states,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            output['loss'] = loss
        
        # Add DiT conditioning for audio generation
        if modality == "audio":
            dit_conditioning = self.audio_decoder_bridge(hidden_states)
            output['dit_conditioning'] = dit_conditioning
        
        return output
    
    def _insert_modality_features(
        self,
        text_hidden: torch.Tensor,
        modality_hidden: torch.Tensor,
        modality_type: str,
    ) -> torch.Tensor:
        """
        Insert modality features into the text sequence.
        Uses special tokens to mark modality boundaries.
        """
        # Simplified: in practice, this uses the model's special token logic
        # For now, concatenate (actual implementation depends on tokenization)
        return torch.cat([text_hidden, modality_hidden], dim=1)
    
    def generate_audio_conditioning(
        self,
        text_prompt: str,
        tokenizer,
    ) -> torch.Tensor:
        """
        Generate conditioning signal for AceStep DiT.
        
        This takes a text prompt and produces the conditioning that
        the AceStep Diffusion Transformer needs to generate music.
        """
        # Tokenize
        inputs = tokenizer(text_prompt, return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(
                input_ids=inputs['input_ids'],
                modality="audio",
            )
        
        return output['dit_conditioning']
    
    def generate_video_conditioning(
        self,
        text_prompt: str,
        tokenizer,
    ) -> torch.Tensor:
        """
        Generate conditioning signal for Cosmos video diffusion.
        """
        inputs = tokenizer(text_prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = self.forward(
                input_ids=inputs['input_ids'],
                modality="vision",
            )
        
        return output['hidden_states']


def load_fused_model(
    fused_weights_path: str,
    config: Optional[OmniStepConfig] = None,
) -> OmniStepModel:
    """
    Load a fused OmniStep model from saved weights.
    
    Args:
        fused_weights_path: Path to fused state dict
        config: Model configuration (uses defaults if None)
    
    Returns:
        Loaded OmniStepModel
    """
    if config is None:
        config = OmniStepConfig()
    
    model = OmniStepModel(config)
    
    # Load fused weights
    state_dict = torch.load(fused_weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded OmniStep model from {fused_weights_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model
