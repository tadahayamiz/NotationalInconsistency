from  .models2 import *

import torch.nn as nn
for cls in [nn.MSELoss, nn.BCEWithLogitsLoss]:
    module_type2class[cls.__name__] = cls

from .modules2.tunnel import *
for cls in [Layer, Tunnel]:
    module_type2class[cls.__name__] = cls

from .modules2.sequence import *
for cls in [TeacherForcer, MaskMaker, SelfAttentionLayer, PositionalEmbedding,
    TransformerEncoder, TransformerDecoder, AttentionDecoder, TransformerLMDecoder,
    GreedyDecoder, CrossEntropyLoss, BCELoss]:
    module_type2class[cls.__name__] = cls

from .modules2.vae import *
for cls in [VAE, MinusD_KLLoss, Random]:
    module_type2class[cls.__name__] = cls

from .modules2.poolers import *
for cls in [MeanPooler, StartPooler, MaxPooler, MeanStartMaxPooler, 
    MeanStartEndMaxPooler, MeanStdStartEndMaxMinPooler, NoAffinePooler, NemotoPooler, 
    GraphPooler]:
    module_type2class[cls.__name__] = cls

from .modules2.graph_transformer import *
for cls in [GraphAttentionLayer, GraphEncoder, AtomEmbedding]:
    module_type2class[cls.__name__] = cls

from .modules2.unimol import *
for cls in [UnimolEncoder, UnimolEncoder2, UnimolGraphEmbedding]:
    module_type2class[cls.__name__] = cls

from .modules2.mlp import *
for cls in [MLP]:
    module_type2class[cls.__name__] = cls
