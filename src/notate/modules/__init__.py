"""Neural network modules: transformers, VAE, poolers, tunnels, and pipeline runner."""

# ---- import concrete module groups ----
from .sequence import *
from .vae import *
from .tunnel import *
from .poolers import *
from .pipeline import PipelineModule  # ★ NEW: pipeline runner (Plan A)

# ---- central registry shared with core ----
from ..core.core import module_type2class

# ---- register common torch losses (convenience) ----
import torch.nn as nn
for cls in [nn.MSELoss, nn.BCEWithLogitsLoss]:
    module_type2class[cls.__name__] = cls

# ---- register sequence modules ----
for cls in [
    TeacherForcer, MaskMaker, SelfAttentionLayer, PositionalEmbedding,
    TransformerEncoder, TransformerDecoder, AttentionDecoder, TransformerLMDecoder,
    GreedyDecoder, CrossEntropyLoss, BCELoss, MLP
]:
    module_type2class[cls.__name__] = cls

# ---- register VAE modules ----
for cls in [VAE, MinusD_KLLoss, Random]:
    module_type2class[cls.__name__] = cls

# ---- register tunnel modules ----
for cls in [Layer, Tunnel]:
    module_type2class[cls.__name__] = cls

# ---- register pooler modules ----
for cls in [
    MeanPooler, StartPooler, MaxPooler, MeanStartMaxPooler,
    MeanStartEndMaxPooler, MeanStdStartEndMaxMinPooler, NoAffinePooler,
    NemotoPooler, GraphPooler
]:
    module_type2class[cls.__name__] = cls

# ---- register pipeline runner (Plan A) ----
module_type2class["PipelineModule"] = PipelineModule  # ★ NEW

__all__ = [
    # Sequence/Transformer modules
    'TeacherForcer', 'MaskMaker', 'SelfAttentionLayer', 'PositionalEmbedding',
    'TransformerEncoder', 'TransformerDecoder', 'AttentionDecoder', 'TransformerLMDecoder',
    'GreedyDecoder', 'CrossEntropyLoss', 'BCELoss', 'MLP',

    # VAE modules
    'VAE', 'MinusD_KLLoss', 'Random',

    # Tunnel modules
    'Layer', 'Tunnel',

    # Pooler modules
    'MeanPooler', 'StartPooler', 'MaxPooler', 'MeanStartMaxPooler',
    'MeanStartEndMaxPooler', 'MeanStdStartEndMaxMinPooler', 'NoAffinePooler',
    'NemotoPooler', 'GraphPooler',

    # Pipeline runner
    'PipelineModule',
]
