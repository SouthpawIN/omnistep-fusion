"""
OmniStep Fusion — Backbone Fusion Engine
Merges Cosmos3-Nano and AceStep transformer weights into unified backbone.

Both models are Qwen3-family with:
- 36 layers, GQA (32Q/8KV), head_dim=128
- SiLU activation, RMSNorm

The merge uses Darwin Family evolution to blend weights where shapes match,
and keeps Cosmos weights for mismatched tensors (larger model = primary).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for backbone fusion."""
    cosmos_path: str        # Path to Cosmos3-Nano model
    acestep_path: str       # Path to AceStep LM model
    output_path: str        # Where to save fused model
    
    # Merge strategy
    merge_ratio: float = 0.6  # Weight toward Cosmos (0.0=AceStep, 1.0=Cosmos)
    shape_match_threshold: float = 0.95  # Minimum shape match ratio to merge
    
    # Projection config
    projection_path: Optional[str] = None  # Path to trained projection weights


class TensorMatcher:
    """
    Matches tensors between two models for merging.
    Handles dimension mismatches via projection.
    """
    
    def __init__(self, cosmos_state: Dict, acestep_state: Dict):
        self.cosmos = cosmos_state
        self.acestep = acestep_state
        self.matches = []
        self.cosmos_only = []
        self.acestep_only = []
    
    def find_matches(self) -> Dict:
        """
        Find matching tensors between models.
        
        Returns:
            Dict with 'matched', 'cosmos_only', 'acestep_only' lists
        """
        cosmos_keys = set(self.cosmos.keys())
        acestep_keys = set(self.acestep.keys())
        
        # Find exact name matches
        exact_matches = cosmos_keys & acestep_keys
        
        for key in exact_matches:
            cosmos_shape = self.cosmos[key].shape
            acestep_shape = self.acestep[key].shape
            
            if cosmos_shape == acestep_shape:
                self.matches.append({
                    'key': key,
                    'type': 'exact',
                    'cosmos_shape': cosmos_shape,
                    'acestep_shape': acestep_shape,
                })
            elif self._shapes_compatible(cosmos_shape, acestep_shape):
                self.matches.append({
                    'key': key,
                    'type': 'projected',
                    'cosmos_shape': cosmos_shape,
                    'acestep_shape': acestep_shape,
                })
            else:
                self.cosmos_only.append(key)
        
        # Find keys only in one model
        self.cosmos_only.extend(cosmos_keys - exact_matches)
        self.acestep_only = list(acestep_keys - exact_matches)
        
        return {
            'matched': self.matches,
            'cosmos_only': self.cosmos_only,
            'acestep_only': self.acestep_only,
        }
    
    def _shapes_compatible(self, shape_a, shape_b) -> bool:
        """Check if two tensor shapes can be bridged via projection."""
        # Same number of dimensions
        if len(shape_a) != len(shape_b):
            return False
        
        # At least one dimension matches (usually the non-hidden dims)
        matching_dims = sum(1 for a, b in zip(shape_a, shape_b) if a == b)
        return matching_dims >= len(shape_a) - 1


class BackboneFusion:
    """
    Fuses two transformer backbones using Darwin merge strategy.
    
    Steps:
    1. Load both models' state dicts
    2. Find matching tensors
    3. For exact matches: weighted blend (Darwin)
    4. For projected matches: project AceStep → Cosmos dimension, then blend
    5. For non-matching: keep Cosmos (primary model)
    6. Save fused state dict
    """
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.cosmos_state = None
        self.acestep_state = None
        self.fused_state = {}
    
    def load_models(self):
        """Load both models' state dicts."""
        print(f"Loading Cosmos from {self.config.cosmos_path}...")
        self.cosmos_state = torch.load(
            self.config.cosmos_path, map_location='cpu', weights_only=True
        )
        if 'state_dict' in self.cosmos_state:
            self.cosmos_state = self.cosmos_state['state_dict']
        
        print(f"Loading AceStep from {self.config.acestep_path}...")
        self.acestep_state = torch.load(
            self.config.acestep_path, map_location='cpu', weights_only=True
        )
        if 'state_dict' in self.acestep_state:
            self.acestep_state = self.acestep_state['state_dict']
        
        print(f"Cosmos: {len(self.cosmos_state)} tensors")
        print(f"AceStep: {len(self.acestep_state)} tensors")
    
    def fuse(self) -> Dict:
        """
        Perform the fusion.
        
        Returns:
            Fused state dict
        """
        self.load_models()
        
        matcher = TensorMatcher(self.cosmos_state, self.acestep_state)
        analysis = matcher.find_matches()
        
        print(f"\nTensor Analysis:")
        print(f"  Exact matches: {len([m for m in analysis['matched'] if m['type'] == 'exact'])}")
        print(f"  Projected matches: {len([m for m in analysis['matched'] if m['type'] == 'projected'])}")
        print(f"  Cosmos only: {len(analysis['cosmos_only'])}")
        print(f"  AceStep only: {len(analysis['acestep_only'])}")
        
        # Process exact matches (weighted blend)
        for match in analysis['matched']:
            if match['type'] == 'exact':
                self._blend_exact(match['key'])
            elif match['type'] == 'projected':
                self._blend_projected(match['key'], match)
        
        # Keep Cosmos-only tensors unchanged
        for key in analysis['cosmos_only']:
            self.fused_state[key] = self.cosmos_state[key]
        
        # Project and add AceStep-only tensors where possible
        for key in analysis['acestep_only']:
            # Only add if it's a transformer layer that could be useful
            if self._should_include_acestep_tensor(key):
                projected = self._project_acestep_tensor(key)
                if projected is not None:
                    self.fused_state[key] = projected
        
        print(f"\nFused model: {len(self.fused_state)} tensors")
        return self.fused_state
    
    def _blend_exact(self, key: str):
        """Blend tensors with exact shape match using Darwin strategy."""
        cosmos_w = self.cosmos_state[key]
        acestep_w = self.acestep_state[key]
        
        # Darwin blend: weighted combination
        alpha = self.config.merge_ratio
        blended = alpha * cosmos_w + (1 - alpha) * acestep_w
        
        self.fused_state[key] = blended
    
    def _blend_projected(self, key: str, match: Dict):
        """Blend tensors with dimension mismatch via projection."""
        cosmos_w = self.cosmos_state[key]
        acestep_w = self.acestep_state[key]
        
        # Project AceStep tensor to Cosmos dimension
        projected = self._project_tensor(acestep_w, match['cosmos_shape'])
        
        if projected is not None:
            # Blend after projection
            alpha = self.config.merge_ratio
            blended = alpha * cosmos_w + (1 - alpha) * projected
            self.fused_state[key] = blended
        else:
            # Projection failed, keep Cosmos
            self.fused_state[key] = cosmos_w
    
    def _project_tensor(self, tensor: torch.Tensor, target_shape: Tuple) -> Optional[torch.Tensor]:
        """
        Project a tensor to match target shape.
        Uses linear projection for the mismatched dimension.
        """
        if tensor.shape == target_shape:
            return tensor
        
        # Find which dimension differs
        mismatched_dims = []
        for i, (a, b) in enumerate(zip(tensor.shape, target_shape)):
            if a != b:
                mismatched_dims.append(i)
        
        if len(mismatched_dims) != 1:
            # Can't project multiple dimensions
            return None
        
        dim = mismatched_dims[0]
        
        # Use linear projection for the mismatched dimension
        if dim == 0:
            # First dim differs (e.g., embedding size)
            proj = nn.Linear(tensor.shape[0], target_shape[0], bias=False)
            # Initialize with identity-like mapping
            with torch.no_grad():
                nn.init.xavier_uniform_(proj.weight)
            return proj(tensor.unsqueeze(0)).squeeze(0)
        elif dim == -1 or dim == len(tensor.shape) - 1:
            # Last dim differs (e.g., hidden size)
            proj = nn.Linear(tensor.shape[-1], target_shape[-1], bias=False)
            with torch.no_grad():
                nn.init.xavier_uniform_(proj.weight)
            return proj(tensor)
        else:
            # Middle dimension - can't easily project
            return None
    
    def _should_include_acestep_tensor(self, key: str) -> bool:
        """Check if an AceStep-only tensor should be included in the fused model."""
        # Include music-specific embeddings and heads
        music_keywords = ['music', 'audio', 'melody', 'beat', 'rhythm', 'note']
        return any(kw in key.lower() for kw in music_keywords)
    
    def _project_acestep_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Project an AceStep tensor to Cosmos dimensions if possible."""
        tensor = self.acestep_state[key]
        
        # Check if it's an embedding that could be resized
        if 'embed' in key.lower() or 'head' in key.lower():
            # These might need vocabulary expansion
            return tensor  # Include as-is for now
        
        return None
    
    def save(self, output_path: Optional[str] = None):
        """Save the fused state dict."""
        if output_path is None:
            output_path = self.config.output_path
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.fused_state, output_path)
        print(f"\nSaved fused model to {output_path}")
        
        # Calculate size
        total_params = sum(p.numel() for p in self.fused_state.values())
        size_gb = sum(p.numel() * p.element_size() for p in self.fused_state.values()) / (1024**3)
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {size_gb:.2f} GB")


# === CLI ===

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fuse Cosmos + AceStep backbones")
    parser.add_argument("--cosmos-path", required=True, help="Path to Cosmos model")
    parser.add_argument("--acestep-path", required=True, help="Path to AceStep LM model")
    parser.add_argument("--output-path", required=True, help="Output path for fused model")
    parser.add_argument("--merge-ratio", type=float, default=0.6, help="Cosmos weight ratio (0-1)")
    
    args = parser.parse_args()
    
    config = FusionConfig(
        cosmos_path=args.cosmos_path,
        acestep_path=args.acestep_path,
        output_path=args.output_path,
        merge_ratio=args.merge_ratio,
    )
    
    fuser = BackboneFusion(config)
    fuser.fuse()
    fuser.save()
