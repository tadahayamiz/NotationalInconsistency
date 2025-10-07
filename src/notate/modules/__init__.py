"""Neural network modules: transformers, VAE, poolers, tunnels, and pipeline runner."""

# ---- import concrete module groups ----
from .sequence import *
from .vae import *
from .tunnel import *
from .poolers import *

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
]


# --- Minimal Affine for scalar/tensor-safe scaling (keeps original config) ---
import torch
import torch.nn as nn
from ..core.core import module_type2class

class Affine(nn.Module):
    """
    y = x * weight + bias
    - weight, bias: Python float でも可（dtype/device は x に自動追従）
    - 任意形状の x に対して PyTorch のブロードキャストで安全に適用
    - 学習対象でない定数変換として使用（論文設定の -d_kl_factor 等）
    """
    def __init__(self, weight: float = 1.0, bias: float = 0.0):
        super().__init__()
        # 学習させない前提なので Parameter 化せず float のまま保持
        self.weight = float(weight)
        self.bias = float(bias)

    def forward(self, x):
        return x * self.weight + self.bias

# register
module_type2class['Affine'] = Affine
__all__.extend([
    'Affine',
])
