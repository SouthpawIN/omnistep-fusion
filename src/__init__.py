"""OmniStep Fusion — Fused omni-modal model."""

from .projection import (
    DimensionProjection,
    BidirectionalProjection,
    RoPEAdapter,
    AudioEncoderBridge,
    AudioDecoderBridge,
    ProjectionConfig,
    ProjectionTrainer,
)

from .vocab_merge import (
    VocabularyMerger,
    VocabMergeConfig,
    resize_embeddings,
)

from .backbone_fusion import (
    BackboneFusion,
    FusionConfig,
    TensorMatcher,
)

from .omnistep_model import (
    OmniStepModel,
    OmniStepConfig,
    load_fused_model,
)

__all__ = [
    # Projection
    'DimensionProjection',
    'BidirectionalProjection',
    'RoPEAdapter',
    'AudioEncoderBridge',
    'AudioDecoderBridge',
    'ProjectionConfig',
    'ProjectionTrainer',
    # Vocabulary
    'VocabularyMerger',
    'VocabMergeConfig',
    'resize_embeddings',
    # Fusion
    'BackboneFusion',
    'FusionConfig',
    'TensorMatcher',
    # Model
    'OmniStepModel',
    'OmniStepConfig',
    'load_fused_model',
]
