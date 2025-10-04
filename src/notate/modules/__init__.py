"""Neural network modules: transformers, VAE, poolers, and tunnels."""

from .sequence import *
from .vae import *  
from .tunnel import *
from .poolers import *

# Register all classes in module_type2class for configuration-based instantiation
from ..core.core import module_type2class

# Import torch.nn for basic loss functions
import torch.nn as nn
for cls in [nn.MSELoss, nn.BCEWithLogitsLoss]:
    module_type2class[cls.__name__] = cls

# Register sequence modules
for cls in [TeacherForcer, MaskMaker, SelfAttentionLayer, PositionalEmbedding,
    TransformerEncoder, TransformerDecoder, AttentionDecoder, TransformerLMDecoder,
    GreedyDecoder, CrossEntropyLoss, BCELoss, MLP]:
    module_type2class[cls.__name__] = cls

# Register VAE modules  
for cls in [VAE, MinusD_KLLoss, Random]:
    module_type2class[cls.__name__] = cls

# Register tunnel modules
for cls in [Layer, Tunnel]:
    module_type2class[cls.__name__] = cls

# Register pooler modules
for cls in [MeanPooler, StartPooler, MaxPooler, MeanStartMaxPooler,
    MeanStartEndMaxPooler, MeanStdStartEndMaxMinPooler, NoAffinePooler,
    NemotoPooler, GraphPooler]:
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
