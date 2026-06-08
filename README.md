# OmniStep Fusion

**One model. All modalities. See, hear, generate video, generate music, think.**

Fuses [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) (vision + video + text) with [AceStep](https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B) (music + audio) into a single unified model.

## What Is This?

OmniStep Fusion creates a truly omni-modal model by:

1. **Merging the backbones** — Both are Qwen3-family with identical transformer internals (GQA 32Q/8KV, head_dim=128, 36 layers). Darwin merge blends the weights.

2. **Bridging dimensions** — Cosmos uses hidden=4096, AceStep uses 2560. Projection layers map between them.

3. **Unifying vocabulary** — 151K Cosmos tokens + 65K music tokens = 217K merged vocabulary.

4. **Attaching all encoders/decoders** — Vision (Cosmos ViT), Audio (AceStep encoder), Video (Cosmos diffusion), Music (AceStep DiT).

## Architecture

```
Vision In ──► [Cosmos ViT] ──► 1152→4096 proj ──┐
                                                  ▼
Audio In ───► [AceStep Encoder] ──► 2560→4096 ──► Backbone ──► Text Out
                                                  (36L, 4096) 
Text In ─────────────────────────────────────────►    │
                                                      ├──► Video Diffusion ──► Video Out
                                                      └──► AceStep DiT ──────► Audio Out
```

## Status

- [x] Architecture design
- [x] Projection layer code
- [x] Vocabulary merger code
- [x] Backbone fusion engine
- [x] Unified model class
- [x] Training recipe
- [ ] Train projection layers (needs parallel embeddings data)
- [ ] Fuse actual model weights (needs Cosmos + AceStep models)
- [ ] Multimodal fine-tuning
- [ ] Benchmarking
- [ ] GGUF quantization

## Quick Start

```bash
# Clone
git clone https://github.com/SouthpawIN/omnistep-fusion.git
cd omnistep-fusion

# Install dependencies
pip install torch transformers

# Run fusion pipeline (requires model downloads)
python3 scripts/fuse_omnistep.py \
    --cosmos-path /path/to/Cosmos3-Nano \
    --acestep-path /path/to/acestep-5Hz-lm-4B \
    --cosmos-tokenizer /path/to/cosmos/tokenizer.json \
    --acestep-tokenizer /path/to/acestep/tokenizer.json \
    --output-dir ./output/ \
    --merge-ratio 0.6
```

## File Structure

```
omnistep-fusion/
├── docs/
│   ├── ARCHITECTURE.md      ← Detailed technical design
│   └── TRAINING.md          ← Training recipes
├── src/
│   ├── projection.py        ← Dimension bridge layers (2560↔4096)
│   ├── vocab_merge.py       ← Vocabulary merger (151K + 65K)
│   ├── backbone_fusion.py   ← Darwin merge engine
│   └── omnistep_model.py    ← Unified model class
├── scripts/
│   └── fuse_omnistep.py     ← Main fusion pipeline
└── README.md
```

## Related Projects

- [Evolutionary Training](https://github.com/SouthpawIN/evolutionary-training) — Darwin merge engine
- [OmniStep 12A3B](https://huggingface.co/sovthpaw/omnistep-12a3b) — Current non-fused version
- [Southpaw's Turbohaul](https://github.com/SouthpawIN/southpaw-turbohaul) — Model server

## License

Apache 2.0 (following parent model licenses)

---

## Related: the OmniSenter architecture

OmniStep-Fusion is one of the pieces of the broader OmniSenter system. The full picture:
- **OmniSenter** (main project): [evolutionary-training](https://github.com/SouthpawIN/evolutionary-training)
- **Design post**: [OmniSenter: The Self-Evolving Multimodal Auxiliary for Hermes](https://github.com/SouthpawIN/evolutionary-training/blob/master/blog/omnisenter-self-evolving.md)
- **Architecture wiki**: `~/wiki/concepts/omnisenter-architecture.md`
- **Cosmos × ACE-Step × Nemotron ASR** master plan: `~/wiki/concepts/omnimodal-fusion-architecture.md`
- **Sparse upcycle** (Stage 3 of the pipeline): [multimodal-expansion](https://github.com/SouthpawIN/multimodal-expansion)

📚 **Master wiki + blog catalog:** [evolutionary-training/wiki](https://github.com/SouthpawIN/evolutionary-training/blob/master/wiki/README.md) — the consolidated knowledge base for the OmniSenter project, in catalog order.
