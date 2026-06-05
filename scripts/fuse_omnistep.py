#!/usr/bin/env python3
"""
OmniStep Fusion — Main Pipeline Script
Orchestrates the full fusion process.

Usage:
    python3 fuse_omnistep.py --cosmos-path /path/to/cosmos --acestep-path /path/to/acestep --output /path/to/output

Steps:
    1. Load and analyze both models
    2. Merge vocabularies
    3. Train dimension projections
    4. Fuse backbones
    5. Attach encoders/decoders
    6. Validate and save
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.projection import ProjectionConfig, DimensionProjection, ProjectionTrainer
from src.vocab_merge import VocabMergeConfig, VocabularyMerger, resize_embeddings
from src.backbone_fusion import FusionConfig, BackboneFusion
from src.omnistep_model import OmniStepConfig, OmniStepModel


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           OmniStep Fusion Pipeline                            ║
║     Cosmos3-Nano + AceStep → One Omni-Modal Model            ║
╚═══════════════════════════════════════════════════════════════╝
""")


def step1_analyze(cosmos_path: str, acestep_path: str):
    """Step 1: Load and analyze both models."""
    print("\n" + "="*60)
    print("STEP 1: Analyzing Models")
    print("="*60)
    
    import torch
    
    # Load state dicts
    print(f"Loading Cosmos from {cosmos_path}...")
    cosmos = torch.load(cosmos_path, map_location='cpu', weights_only=True)
    if 'state_dict' in cosmos:
        cosmos = cosmos['state_dict']
    
    print(f"Loading AceStep from {acestep_path}...")
    acestep = torch.load(acestep_path, map_location='cpu', weights_only=True)
    if 'state_dict' in acestep:
        acestep = acestep['state_dict']
    
    # Analyze
    cosmos_params = sum(p.numel() for p in cosmos.values())
    acestep_params = sum(p.numel() for p in acestep.values())
    
    print(f"\nCosmos3-Nano:")
    print(f"  Tensors: {len(cosmos)}")
    print(f"  Parameters: {cosmos_params:,}")
    print(f"  Size: {sum(p.numel() * p.element_size() for p in cosmos.values()) / (1024**3):.2f} GB")
    
    print(f"\nAceStep LM:")
    print(f"  Tensors: {len(acestep)}")
    print(f"  Parameters: {acestep_params:,}")
    print(f"  Size: {sum(p.numel() * p.element_size() for p in acestep.values()) / (1024**3):.2f} GB")
    
    # Find matching tensors
    cosmos_keys = set(cosmos.keys())
    acestep_keys = set(acestep.keys())
    common = cosmos_keys & acestep_keys
    
    exact_matches = []
    projected_matches = []
    for key in common:
        if cosmos[key].shape == acestep[key].shape:
            exact_matches.append(key)
        else:
            projected_matches.append(key)
    
    print(f"\nTensor Matching:")
    print(f"  Exact matches: {len(exact_matches)}")
    print(f"  Shape mismatches: {len(projected_matches)}")
    print(f"  Cosmos only: {len(cosmos_keys - common)}")
    print(f"  AceStep only: {len(acestep_keys - common)}")
    
    return cosmos, acestep, exact_matches, projected_matches


def step2_merge_vocab(cosmos_tokenizer: str, acestep_tokenizer: str, output_dir: str):
    """Step 2: Merge vocabularies."""
    print("\n" + "="*60)
    print("STEP 2: Merging Vocabularies")
    print("="*60)
    
    config = VocabMergeConfig(
        cosmos_vocab_path=cosmos_tokenizer,
        acestep_vocab_path=acestep_tokenizer,
        output_dir=output_dir,
    )
    
    merger = VocabularyMerger(config)
    merged_vocab = merger.merge()
    merger.save_merged_tokenizer()
    
    mapping = merger.create_token_id_mapping()
    mapping_path = os.path.join(output_dir, 'token_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nMerged vocabulary: {len(merged_vocab)} tokens")
    print(f"Music tokens added: {len(merged_vocab) - merger.music_token_offset}")
    
    return mapping


def step3_train_projections(cosmos_state: dict, acestep_state: dict, output_dir: str):
    """Step 3: Train dimension projection layers."""
    print("\n" + "="*60)
    print("STEP 3: Training Projection Layers")
    print("="*60)
    
    import torch
    
    # Extract hidden states from both models for projection training
    # In practice, you'd generate these from actual model inference
    # For now, we initialize with random data as a starting point
    
    print("Initializing projection layers...")
    
    # Audio encoder bridge: 2560 → 4096
    audio_bridge = DimensionProjection(ProjectionConfig(
        input_dim=2560, output_dim=4096, num_layers=2
    ))
    
    # Audio decoder bridge: 4096 → 1024
    audio_decoder_bridge = DimensionProjection(ProjectionConfig(
        input_dim=4096, output_dim=1024, num_layers=2
    ))
    
    # Vision projection: 1152 → 4096 (Cosmos already has this)
    vision_proj = DimensionProjection(ProjectionConfig(
        input_dim=1152, output_dim=4096, num_layers=1
    ))
    
    # Save projection weights
    proj_path = os.path.join(output_dir, 'projections.pt')
    torch.save({
        'audio_encoder_bridge': audio_bridge.state_dict(),
        'audio_decoder_bridge': audio_decoder_bridge.state_dict(),
        'vision_projection': vision_proj.state_dict(),
    }, proj_path)
    
    print(f"Saved projection layers to {proj_path}")
    print("NOTE: These are initialized with random weights.")
    print("Train on parallel embeddings before using for fusion.")
    
    return {
        'audio_encoder_bridge': audio_bridge,
        'audio_decoder_bridge': audio_decoder_bridge,
        'vision_projection': vision_proj,
    }


def step4_fuse_backbones(cosmos_state: dict, acestep_state: dict, output_dir: str, merge_ratio: float = 0.6):
    """Step 4: Fuse the transformer backbones."""
    print("\n" + "="*60)
    print("STEP 4: Fusing Backbones")
    print("="*60)
    
    fused_path = os.path.join(output_dir, 'fused_backbone.pt')
    
    config = FusionConfig(
        cosmos_path="",  # We already have the state dict
        acestep_path="",
        output_path=fused_path,
        merge_ratio=merge_ratio,
    )
    
    fuser = BackboneFusion(config)
    fuser.cosmos_state = cosmos_state
    fuser.acestep_state = acestep_state
    fused_state = fuser.fuse()
    
    # Save fused backbone
    torch.save(fused_state, fused_path)
    
    total_params = sum(p.numel() for p in fused_state.values())
    size_gb = sum(p.numel() * p.element_size() for p in fused_state.values()) / (1024**3)
    
    print(f"\nFused backbone:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"  Saved to: {fused_path}")
    
    return fused_state


def step5_assemble_model(fused_state: dict, projections: dict, mapping: dict, output_dir: str):
    """Step 5: Assemble the complete model."""
    print("\n" + "="*60)
    print("STEP 5: Assembling OmniStep Model")
    print("="*60)
    
    config = OmniStepConfig()
    model = OmniStepModel(config)
    
    # Load fused backbone weights
    model.load_state_dict(fused_state, strict=False)
    
    # Load projection weights
    model.audio_encoder_bridge.load_state_dict(projections['audio_encoder_bridge'].state_dict())
    model.audio_decoder_bridge.load_state_dict(projections['audio_decoder_bridge'].state_dict())
    model.vision_projection.load_state_dict(projections['vision_projection'].state_dict())
    
    # Resize embeddings for merged vocabulary
    new_vocab_size = config.vocab_size
    old_vocab_size = 151936  # Cosmos original vocab size
    
    # Resize with mean initialization for new music tokens
    model = resize_embeddings(
        model,
        new_vocab_size=new_vocab_size,
        old_vocab_size=old_vocab_size,
        token_mapping=mapping.get('cosmos_mapping', {}),
        init_strategy="mean",
    )
    
    # Save complete model
    model_path = os.path.join(output_dir, 'omnistep_fused.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    import dataclasses
    with open(config_path, 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nOmniStep model assembled:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Saved to: {model_path}")
    print(f"  Config: {config_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="OmniStep Fusion Pipeline")
    parser.add_argument("--cosmos-path", required=True, help="Path to Cosmos3-Nano model")
    parser.add_argument("--acestep-path", required=True, help="Path to AceStep LM model")
    parser.add_argument("--cosmos-tokenizer", required=True, help="Path to Cosmos tokenizer.json")
    parser.add_argument("--acestep-tokenizer", required=True, help="Path to AceStep tokenizer.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--merge-ratio", type=float, default=0.6, help="Cosmos weight ratio (0-1)")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Analyze
    cosmos, acestep, exact_matches, projected_matches = step1_analyze(
        args.cosmos_path, args.acestep_path
    )
    
    # Step 2: Merge vocabularies
    mapping = step2_merge_vocab(
        args.cosmos_tokenizer, args.acestep_tokenizer, args.output_dir
    )
    
    # Step 3: Train projections
    projections = step3_train_projections(cosmos, acestep, args.output_dir)
    
    # Step 4: Fuse backbones
    fused_state = step4_fuse_backbones(cosmos, acestep, args.output_dir, args.merge_ratio)
    
    # Step 5: Assemble
    model = step5_assemble_model(fused_state, projections, mapping, args.output_dir)
    
    print("\n" + "="*60)
    print("FUSION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Train projections on parallel embeddings")
    print(f"  2. Fine-tune model on multimodal data")
    print(f"  3. Quantize to GGUF for deployment")
    print(f"  4. Benchmark all modalities")


if __name__ == "__main__":
    main()
