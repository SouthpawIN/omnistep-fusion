# OmniStep Fusion — Training Recipe

## Overview

Training the fused OmniStep model happens in 3 phases:

1. **Projection Training** (fast, ~hours)
2. **Multimodal Fine-Tuning** (medium, ~days)
3. **Capability Distillation** (slow, ~weeks)

---

## Phase 1: Projection Training

**Goal:** Train the dimension bridge layers (2560↔4096) so that
AceStep's audio representations can flow into Cosmos's backbone.

**Data:** Parallel text/audio pairs from both models

### Steps

```bash
# 1. Generate parallel embeddings
python3 scripts/generate_parallel_embeddings.py \
    --cosmos-path /path/to/cosmos \
    --acestep-path /path/to/acestep \
    --data-path /path/to/text-audio-pairs \
    --output-path training_data/parallel_embeddings.pt

# 2. Train projection layers
python3 scripts/train_projections.py \
    --embeddings training_data/parallel_embeddings.pt \
    --output-dir projections/ \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4

# 3. Validate projections
python3 scripts/validate_projections.py \
    --projections projections/best.pt \
    --test-data training_data/test_embeddings.pt
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 | AdamW, cosine schedule |
| Batch size | 32 | Gradient accumulation if needed |
| Epochs | 100 | Early stopping on validation loss |
| Loss | 0.5×cosine + 0.3×MSE + 0.2×task | See ProjectionTrainer |
| Warmup | 10 epochs | Linear warmup |

### Expected Results

- Cosine similarity: >0.95 between projected and target embeddings
- MSE loss: <0.01
- Task loss: Should decrease steadily

---

## Phase 2: Multimodal Fine-Tuning

**Goal:** Train the fused model to handle all modalities together.

**Data:** Mixed dataset of text, images, video, music, audio

### Dataset Structure

```
training_data/
├── text/                    # Text generation data
│   ├── instructions.jsonl
│   └── conversations.jsonl
├── vision/                  # Image understanding
│   ├── captions.jsonl
│   └── vqa.jsonl
├── video/                   # Video understanding
│   ├── descriptions.jsonl
│   └── qa.jsonl
├── music/                   # Music understanding + generation
│   ├── descriptions.jsonl
│   ├── lyrics.jsonl
│   └── midi_analysis.jsonl
└── mixed/                   # Cross-modal data
    ├── music_video.jsonl    # Music + video pairs
    └── text_audio.jsonl     # Text + audio pairs
```

### Training Script

```bash
python3 training/sft_multimodal.py \
    --model-path /path/to/assembled_omnistep \
    --data-path training_data/ \
    --output-dir checkpoints/ \
    --epochs 10 \
    --batch-size 4 \
    --lr 2e-5 \
    --gradient-accumulation 8 \
    --warmup-steps 1000 \
    --save-steps 500 \
    --eval-steps 100
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 2e-5 | AdamW, cosine schedule |
| Batch size | 4 | With gradient accumulation 8 = effective 32 |
| Epochs | 10 | More for better quality |
| Warmup | 1000 steps | Linear warmup |
| Max sequence | 4096 | Truncate longer sequences |
| Gradient clip | 1.0 | Prevent exploding gradients |

### Progressive Unfreezing

```
Epoch 1-2:   Freeze backbone, train projections only
Epoch 3-5:   Unfreeze last 6 layers of backbone
Epoch 6-8:   Unfreeze all backbone layers
Epoch 9-10:  Unfreeze everything (including encoders/decoders)
```

---

## Phase 3: Capability Distillation (Optional)

**Goal:** Recover any capabilities lost during fusion.

If benchmarks show degradation in specific modalities:

### Music Distillation

```bash
# Distill music knowledge from original AceStep
python3 training/distill_music.py \
    --teacher-path /path/to/acestep \
    --student-path checkpoints/best/ \
    --data-path training_data/music/ \
    --output-dir checkpoints/music_distilled/ \
    --distill-weight 0.3
```

### Vision Distillation

```bash
# Distill vision knowledge from original Cosmos
python3 training/distill_vision.py \
    --teacher-path /path/to/cosmos \
    --student-path checkpoints/best/ \
    --data-path training_data/vision/ \
    --output-dir checkpoints/vision_distilled/ \
    --distill-weight 0.3
```

---

## Data Requirements

### Minimum for Basic Functionality

| Modality | Samples | Source |
|----------|---------|--------|
| Text | 100K | OpenAssistant, ShareGPT |
| Vision | 50K | LLaVA, COCO |
| Video | 10K | WebVid, ActivityNet |
| Music | 50K | MusicCaps, NSynth |
| Mixed | 10K | Custom pairs |

### Recommended for Production

| Modality | Samples | Source |
|----------|---------|--------|
| Text | 1M+ | Hermes agent traces, web data |
| Vision | 500K+ | Multi-modal datasets |
| Video | 100K+ | Video understanding datasets |
| Music | 200K+ | Music + lyrics datasets |
| Mixed | 50K+ | Custom cross-modal pairs |

---

## Monitoring

### Key Metrics to Track

1. **Per-modality loss** — should decrease for all modalities
2. **Cross-modal retrieval** — can model match music↔video?
3. **Generation quality** — BLEU for text, FID for images, FAD for audio
4. **Catastrophic forgetting** — compare to baseline models

### Recommended Tools

- Weights & Biases for experiment tracking
- TensorBoard for real-time loss curves
- Custom evaluation scripts per modality

---

## Hardware Requirements

| Phase | GPU | Time | Notes |
|-------|-----|------|-------|
| Projection training | 1× RTX 3090 | 2-4 hours | Small model, fast |
| Multimodal fine-tuning | 2-4× RTX 3090 | 3-7 days | Full model training |
| Distillation | 2× RTX 3090 | 2-5 days | Per modality |

### Optimization

- Use gradient checkpointing for memory efficiency
- Use mixed precision (bf16) for speed
- Use DeepSpeed ZeRO-3 for multi-GPU
- Use flash attention for efficiency

---

## Checkpoint Management

```
checkpoints/
├── projection_best.pt         # Best projection weights
├── epoch_01/                  # Full model checkpoint
├── epoch_02/
├── ...
├── best/                      # Best overall checkpoint
└── final/                     # Final checkpoint
```

---

## Post-Training

After training completes:

1. **Benchmark** all modalities against baseline models
2. **Quantize** to GGUF (Q4_K_M for deployment)
3. **Test** end-to-end with real inputs
4. **Deploy** via llama-server or OmniStep inference engine
5. **Upload** to HuggingFace

```bash
# Quantize to GGUF
python3 scripts/quantize.py \
    --model-path checkpoints/best/ \
    --output-path omnistep-fused-Q4_K_M.gguf \
    --quant-type Q4_K_M
```
