"""
OmniStep Fusion — Vocabulary Merger
Merges Cosmos vocab (151K) + AceStep music tokens (~65K) into unified vocabulary.

This creates a single tokenizer that can handle:
- Standard text tokens (shared between both models)
- Vision tokens (from Cosmos: image_token, video_token, vision_start/end)
- Music tokens (from AceStep: audio features, beat patterns, notation)
"""

import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class VocabMergeConfig:
    """Configuration for vocabulary merging."""
    cosmos_vocab_path: str      # Path to Cosmos tokenizer.json
    acestep_vocab_path: str     # Path to AceStep tokenizer.json
    output_dir: str             # Where to save merged tokenizer
    conflict_strategy: str = "cosmos_wins"  # How to handle duplicate tokens
    min_music_token_freq: int = 0  # Minimum frequency for music tokens to include


class VocabularyMerger:
    """
    Merges two tokenizers into one unified vocabulary.
    
    Strategy:
    1. Take Cosmos vocab as base (151,936 tokens)
    2. Identify AceStep tokens not in Cosmos
    3. Add music-specific tokens to the end of Cosmos vocab
    4. Create merged tokenizer with all tokens
    5. Update token IDs for both models
    """
    
    def __init__(self, config: VocabMergeConfig):
        self.config = config
        self.cosmos_vocab = {}
        self.acestep_vocab = {}
        self.merged_vocab = {}
        self.music_token_offset = 0
    
    def load_tokenizers(self):
        """Load both tokenizer vocabularies."""
        print("Loading Cosmos tokenizer...")
        with open(self.config.cosmos_vocab_path, 'r') as f:
            cosmos_data = json.load(f)
            self.cosmos_vocab = {v[0]: k for k, v in cosmos_data.get('model', {}).get('vocab', {}).items()}
        
        print("Loading AceStep tokenizer...")
        with open(self.config.acestep_vocab_path, 'r') as f:
            acestep_data = json.load(f)
            self.acestep_vocab = {v[0]: k for k, v in acestep_data.get('model', {}).get('vocab', {}).items()}
        
        print(f"Cosmos vocab: {len(self.cosmos_vocab)} tokens")
        print(f"AceStep vocab: {len(self.acestep_vocab)} tokens")
    
    def find_music_tokens(self) -> list:
        """
        Identify AceStep tokens that are music-specific.
        These are tokens NOT in Cosmos's vocabulary.
        """
        cosmos_tokens = set(self.cosmos_vocab.values())
        acestep_tokens = set(self.acestep_vocab.values())
        
        # Music tokens = AceStep tokens not in Cosmos
        music_tokens = []
        for token in acestep_tokens:
            if token not in cosmos_tokens:
                # Check if it looks like a music token (not just a different encoding of same text)
                if self._is_music_token(token):
                    music_tokens.append(token)
        
        print(f"Found {len(music_tokens)} music-specific tokens")
        return music_tokens
    
    def _is_music_token(self, token: str) -> bool:
        """
        Heuristic to identify music-specific tokens.
        Music tokens often contain special characters or patterns.
        """
        # Skip very common text tokens that might just be encoded differently
        if len(token) <= 2 and token.isalpha():
            return False
        
        # Music tokens often have these patterns
        music_indicators = ['♪', '♫', '♬', '♭', '♯', '#', '@', '<', '>']
        
        # Check for music-specific patterns
        if any(indicator in token for indicator in music_indicators):
            return True
        
        # Check for numeric patterns common in audio features
        if token.isdigit() and len(token) >= 3:
            return True
        
        # Check for special prefix patterns used in music tokens
        if token.startswith(('▁', '<', '[', '{')):
            return True
        
        return True  # Default: include as potential music token
    
    def merge(self) -> dict:
        """
        Merge the two vocabularies.
        
        Returns:
            Merged vocabulary dict mapping token → id
        """
        self.load_tokenizers()
        music_tokens = self.find_music_tokens()
        
        # Start with Cosmos vocab (base)
        self.merged_vocab = dict(self.cosmos_vocab)
        base_size = len(self.merged_vocab)
        
        # Add music tokens at the end
        for i, token in enumerate(music_tokens):
            new_id = base_size + i
            self.merged_vocab[token] = new_id
        
        self.music_token_offset = base_size
        total_size = len(self.merged_vocab)
        
        print(f"\nMerged vocabulary:")
        print(f"  Base (Cosmos): {base_size}")
        print(f"  Music tokens:  {len(music_tokens)}")
        print(f"  Total:         {total_size}")
        print(f"  Music offset:  {self.music_token_offset}")
        
        return self.merged_vocab
    
    def create_token_id_mapping(self) -> dict:
        """
        Create mapping from old token IDs to new merged IDs.
        Useful for resizing embedding layers.
        
        Returns:
            Dict with 'cosmos_mapping' and 'acestep_mapping'
        """
        cosmos_mapping = {}
        for token, old_id in self.cosmos_vocab.items():
            cosmos_mapping[old_id] = self.merged_vocab.get(token, old_id)
        
        acestep_mapping = {}
        for token, old_id in self.acestep_vocab.items():
            if token in self.merged_vocab:
                acestep_mapping[old_id] = self.merged_vocab[token]
            else:
                acestep_mapping[old_id] = None  # Token not in merged vocab
        
        return {
            'cosmos_mapping': cosmos_mapping,
            'acestep_mapping': acestep_mapping,
            'music_token_offset': self.music_token_offset,
        }
    
    def save_merged_tokenizer(self, output_path: Optional[str] = None):
        """Save the merged tokenizer to disk."""
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, 'merged_tokenizer.json')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create tokenizer.json format
        merged_tokenizer = {
            'version': '1.0',
            'model': {
                'type': 'BPE',
                'vocab': {token: idx for idx, token in enumerate(self.merged_vocab.keys())},
            },
            'metadata': {
                'source': 'OmniStep Fusion',
                'cosmos_vocab_size': len(self.cosmos_vocab),
                'acestep_vocab_size': len(self.acestep_vocab),
                'music_token_offset': self.music_token_offset,
                'total_vocab_size': len(self.merged_vocab),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(merged_tokenizer, f, indent=2)
        
        print(f"Saved merged tokenizer to {output_path}")
        return output_path


def resize_embeddings(
    model,
    new_vocab_size: int,
    old_vocab_size: int,
    token_mapping: dict,
    init_strategy: str = "mean"
):
    """
    Resize model embedding layers for new vocabulary.
    
    Args:
        model: The model with embedding layers to resize
        new_vocab_size: New vocabulary size
        old_vocab_size: Old vocabulary size
        token_mapping: Dict mapping old IDs to new IDs
        init_strategy: How to initialize new embeddings ("mean", "random", "zero")
    """
    import torch
    
    # Get current embeddings
    old_embed = model.get_input_embeddings()
    old_weight = old_embed.weight.data
    
    # Create new embedding layer
    new_embed = torch.nn.Embedding(new_vocab_size, old_weight.shape[1])
    new_weight = new_embed.weight.data
    
    # Copy existing embeddings
    for old_id, new_id in token_mapping.items():
        if new_id is not None and old_id < old_weight.shape[0]:
            new_weight[new_id] = old_weight[old_id]
    
    # Initialize new tokens
    if init_strategy == "mean":
        # Use mean of existing embeddings
        mean_embed = old_weight.mean(dim=0)
        for i in range(new_vocab_size):
            if i not in token_mapping.values():
                new_weight[i] = mean_embed
    elif init_strategy == "random":
        # Random initialization
        for i in range(new_vocab_size):
            if i not in token_mapping.values():
                new_weight[i] = torch.randn_like(old_weight[0]) * 0.02
    
    # Replace embedding layer
    model.set_input_embeddings(new_embed)
    
    # Also resize LM head if it exists
    if hasattr(model, 'lm_head'):
        old_lm = model.lm_head
        new_lm = torch.nn.Linear(old_lm.weight.shape[1], new_vocab_size, bias=False)
        new_lm.weight.data = new_weight
        model.lm_head = new_lm
    
    print(f"Resized embeddings: {old_vocab_size} → {new_vocab_size}")
    return model


# === CLI ===

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge Cosmos + AceStep vocabularies")
    parser.add_argument("--cosmos-tokenizer", required=True, help="Path to Cosmos tokenizer.json")
    parser.add_argument("--acestep-tokenizer", required=True, help="Path to AceStep tokenizer.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged tokenizer")
    
    args = parser.parse_args()
    
    config = VocabMergeConfig(
        cosmos_vocab_path=args.cosmos_tokenizer,
        acestep_vocab_path=args.acestep_tokenizer,
        output_dir=args.output_dir,
    )
    
    merger = VocabularyMerger(config)
    merger.merge()
    merger.save_merged_tokenizer()
    
    mapping = merger.create_token_id_mapping()
    mapping_path = os.path.join(args.output_dir, 'token_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved token mapping to {mapping_path}")
