# OmniStep Fusion: The Complete Architecture
## Fusing Cosmos3-Nano + AceStep into One Omni-Modal Model

**Date:** 2026-06-05
**Author:** Southpaw / Nous Research Community
**Status:** Design + Prototype Phase

---

## Executive Summary

Fuse Cosmos3-Nano (16B, vision+video+text) with AceStep (4B, music+audio) into a
single unified model that can **see, hear, generate video, generate music, and think**.

Both models are Qwen3-family with identical transformer internals (GQA 32Q/8KV,
head_dim=128, 36 layers, SiLU, RMSNorm). The core "brain" speaks the same language.
We bridge the dimension gap (4096↔2560) with projection layers and expand the
vocabulary to include music tokens.

---

## Architecture Comparison

| Property | Cosmos3-Nano | AceStep LM | AceStep DiT | Compatible? |
|----------|-------------|------------|-------------|-------------|
| **Architecture** | Qwen3-VL (MoT) | Qwen3-4B | AceStep DiT | ✅ Qwen3 family |
| **hidden_size** | 4,096 | 2,560 | 2,048 | ❌ Need projection |
| **num_layers** | 36 | 36 | 24 | ✅ LMs match |
| **num_heads** | 32 | 32 | 16 | ✅ LMs match |
| **num_kv_heads** | 8 | 8 | 8 | ✅ Match |
| **head_dim** | 128 | 128 | 128 | ✅ Match |
| **vocab_size** | 151,936 | 217,204 | 64,003 | ❌ Need merge |
| **RoPE** | M-RoPE (5M θ) | Standard (1M θ) | Standard (1M θ) | ❌ Need adapter |
| **Activation** | SiLU | SiLU | SiLU | ✅ Match |
| **Norm** | RMSNorm 1e-6 | RMSNorm 1e-6 | RMSNorm 1e-6 | ✅ Match |

### What's Already Compatible (No Changes Needed)
- Transformer block structure (attention + FFN layout)
- GQA pattern (32 query heads, 8 KV heads)
- Head dimension (128)
- Layer count (36)
- Activation function (SiLU)
- Normalization (RMSNorm)

### What Needs Bridging
1. **Hidden dimension**: 2560 → 4096 (AceStep → Cosmos)
2. **RoPE encoding**: Standard → M-RoPE (or unified adapter)
3. **Vocabulary**: 151K + 65K music tokens = ~217K merged
4. **Encoder interfaces**: Vision (Cosmos) + Audio (AceStep) → shared backbone
5. **Decoder interfaces**: Video diffusion (Cosmos) + Audio diffusion (AceStep)

---

## Fusion Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT MODALITIES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Vision ──► [Qwen3-VL ViT] ──► 1152→4096 projection ──┐        │
│             (27 layers, 1152 hidden)                    │        │
│                                                         ▼        │
│  Audio ───► [AceStep Audio Encoder] ──► 2560→4096 ──► CONCAT    │
│             (4 layers, timbre encoder)    projection     │        │
│                                                         │        │
│  Text ────► [Tokenizer + Embedding] ─────────────────►  │        │
│             (merged vocab: 151K + 65K music)             │        │
│                                                         ▼        │
├─────────────────────────────────────────────────────────────────┤
│                    SHARED BACKBONE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Qwen3 Transformer (36 layers)                           │   │
│  │  hidden=4096, heads=32, kv_heads=8, head_dim=128        │   │
│  │  Merged weights: Cosmos + AceStep (Darwin blend)         │   │
│  │  Unified RoPE adapter (M-RoPE compatible)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
├──────────────────────────────┼────────────────────────────────────┤
│                    OUTPUT HEADS                                  │
├──────────────────────────────┼────────────────────────────────────┤
│                              │                                    │
│                              ├──► Text Head ──► Text Output       │
│                              │    (merged vocab, 217K tokens)     │
│                              │                                    │
│                              ├──► [4096→1024 projection]          │
│                              │    ──► AceStep DiT ──► Audio Out   │
│                              │    (24 layers, 2048 hidden)        │
│                              │    + Audio Decoder (24 layers)     │
│                              │    + Lyric Encoder (8 layers)      │
│                              │    + Timbre Encoder (4 layers)     │
│                              │                                    │
│                              └──► Cosmos Diffusion Expert         │
│                                   ──► Video/Image Out             │
│                                   + Wan2.2 VAE decoder            │
│                                   + Sound Tokenizer (AVAE)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Vocabulary Merge
**Goal:** Create a unified tokenizer that handles text + music tokens

- Take Cosmos vocab (151,936 tokens) as base
- Add AceStep's music-specific tokens (~65K tokens) that aren't in Cosmos
- Result: ~217K merged vocabulary
- Resize embedding layers and LM head accordingly

### Phase 2: Dimension Projection
**Goal:** Bridge 2560-dim (AceStep) ↔ 4096-dim (Cosmos)

- Train a **2560→4096 linear projection** + LayerNorm
- Training data: parallel text embeddings from both models
- Loss: cosine similarity + MSE between projected and target embeddings
- This projection maps AceStep's "music understanding" into Cosmos's space

### Phase 3: Backbone Fusion (Darwin Merge)
**Goal:** Merge the transformer weights

- Both have 36 layers with same GQA structure
- Darwin merge on matching tensor shapes (most will match after projection)
- For mismatched tensors: keep Cosmos weights (larger model)
- Result: one backbone that has knowledge from both models

### Phase 4: Encoder Attachment
**Goal:** Connect both sets of encoders

- Cosmos Vision Encoder: already attached (1152→4096 projection)
- AceStep Audio Encoder: attach via new 2560→4096 projection layer
- Train projection on paired audio-text data

### Phase 5: Decoder Attachment
**Goal:** Connect both sets of decoders/generators

- Cosmos Diffusion Expert: already attached (video/image generation)
- AceStep DiT: attach via new 4096→1024 projection (backbone→DiT conditioning)
- Keep all DiT sub-components: Audio Decoder, Lyric Encoder, Timbre Encoder

### Phase 6: Multimodal Fine-Tuning
**Goal:** Train the fused model on mixed modalities

- Dataset: text, images, video, music, audio
- Progressive unfreezing: projections first, then backbone, then decoders
- Modalities: text generation, image understanding, video understanding,
  music understanding, music generation, video generation

---

## Technical Details

### RoPE Reconciliation
Cosmos uses M-RoPE (multimodal, sections [24,20,20], θ=5M)
AceStep uses standard RoPE (θ=1M)

**Solution:** Train a RoPE adapter that:
1. Uses M-RoPE as the base (supports both text and spatial positions)
2. Maps AceStep's standard positions into M-RoPE's section layout
3. Fine-tune with mixed modality data

### Vocabulary Merge Strategy
```
Cosmos vocab:  151,936 tokens (text + vision tokens)
AceStep vocab: 217,204 tokens (text + 65K music tokens)
Overlap:       ~151K tokens (shared text vocabulary)

Merged vocab:  151,936 (Cosmos base) + ~65K music tokens = ~217K
New tokens:    Music notation, audio features, beat patterns, etc.
```

### Projection Layer Architecture
```python
class DimensionProjection(nn.Module):
    """Bridge 2560-dim (AceStep) → 4096-dim (Cosmos)"""
    def __init__(self, input_dim=2560, output_dim=4096):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.RMSNorm(output_dim, eps=1e-6)
    
    def forward(self, x):
        return self.norm(self.proj(x))
```

---

## File Structure

```
omnistep-fusion/
├── docs/
│   ├── ARCHITECTURE.md          ← This file
│   ├── TRAINING.md              ← Training recipes
│   └── RESULTS.md               ← Benchmark results
├── configs/
│   ├── fusion_config.yaml       ← Model configuration
│   └── training_config.yaml     ← Training hyperparameters
├── src/
│   ├── projection.py            ← Dimension projection layers
│   ├── vocab_merge.py           ← Vocabulary merger
│   ├── backbone_fusion.py       ← Darwin merge engine
│   ├── encoder_bridge.py        ← Encoder attachment
│   ├── decoder_bridge.py        ← Decoder attachment
│   └── omnistep_model.py        ← Unified model class
├── scripts/
│   ├── prepare_projections.py   ← Train projection layers
│   ├── fuse_backbone.py         ← Run backbone fusion
│   ├── attach_encoders.py       ← Connect encoders
│   ├── attach_decoders.py       ← Connect decoders
│   └── benchmark.py             ← Test all modalities
└── training/
    ├── sft_multimodal.py        ← Multimodal fine-tuning
    └── data/                    ← Training data manifests
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Projection layers don't converge | Medium | Use larger projection (2-layer MLP instead of linear) |
| Merged backbone loses capabilities | Medium | Conservative merge ratios, benchmark each step |
| RoPE incompatibility causes position errors | High | Train RoPE adapter with position-annotated data |
| Music generation quality drops | Medium | Keep AceStep DiT mostly frozen, only train bridge |
| Video generation quality drops | Low | Keep Cosmos diffusion expert frozen |
| Model too large for target hardware | Medium | Quantize (GGUF Q4_K_M fits on 24GB GPU) |

---

## Success Criteria

1. **Text**: Merged model maintains Cosmos's text quality (MMLU within 2%)
2. **Vision**: Image understanding unchanged (frozen encoder)
3. **Video**: Video generation quality unchanged (frozen diffusion expert)
4. **Music**: Music generation quality within 10% of standalone AceStep
5. **Size**: GGUF Q4_K_M fits on single 24GB GPU (~16GB)
6. **Speed**: Inference speed comparable to Cosmos3-Nano alone

---

## References

- Cosmos3-Nano: nvidia/Cosmos3-Nano (HuggingFace)
- ACE-Step: ACE-Step/acestep-5Hz-lm-4B (HuggingFace)
- Darwin Family Merge: evolutionary-training/scripts/qwen_cosmos_darwin_merge.py
- OmniStep 12A3B: sovthpaw/omnistep-12a3b (current non-fused version)
