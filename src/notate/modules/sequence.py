#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/modules/sequence.py

Purpose:
    Sequence-related modules:
      - Teacher forcing / mask makers
      - Positional encoding & embedding
      - Transformer-based encoder/decoders (including step-wise decoding)
      - Greedy/beam decoding helpers
      - Loss wrappers

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports ordered: stdlib -> third-party -> local (project).
    - No behavior-changing edits were made intentionally.
    - Activation handling keeps original semantics; string names map to torch/nn.F.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import copy
import math
import sys

# ===== Third-party =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ===== Project-local =====
from ..core.core import function_config2func, init_config2func
from ..core import register_module

# =============================================================================
# Activation mapping
# =============================================================================
ACTIVATION_FUNCTIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "swish": lambda x: x * torch.sigmoid(x),
    "silu": F.silu,
}


# =============================================================================
# Utility modules
# =============================================================================
class TeacherForcer(nn.Module):
    """Create teacher-forcing input/target pairs by shifting the sequence."""

    def __init__(self, length_dim: int):
        super().__init__()
        self.length_dim = length_dim
        self.input_slices = [slice(None)] * length_dim + [slice(None, -1)]
        self.target_slices = [slice(None)] * length_dim + [slice(1, None)]

    def forward(self, input, return_len: bool = False):
        """Split into (input, target) shifted by one along length dimension.

        Args:
            input: Tensor-like with a length dimension.
            return_len: If True, also return the resulting sequence length.

        Returns:
            Tuple (input_shifted, target_shifted[, length]).
        """
        out = (input[tuple(self.input_slices)], input[tuple(self.target_slices)])
        if return_len:
            out += (out[-1].shape[self.length_dim],)
        return out


class MaskMaker(nn.Module):
    """Create equality/inequality masks against a specific token id."""

    def __init__(self, mask_token, dtype: str = "bool", direction: str = "equal"):
        super().__init__()
        self.mask_token = mask_token
        self.dtype = dtype
        self.direction = direction

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return mask with dtype in {'bool','int'}.

        Args:
            input: Integer tensor.

        Returns:
            Mask tensor where True/1 indicates a match (or mismatch if direction != 'equal').
        """
        if self.direction == "equal":
            mask = input == self.mask_token
        else:
            mask = input != self.mask_token

        if self.dtype == "bool":
            return mask
        if self.dtype == "int":
            return mask.to(torch.int)
        return mask


# =============================================================================
# Transformer encoder (with attention weight option)
# =============================================================================
class SelfAttentionLayer_old(nn.TransformerEncoderLayer):
    """nn.TransformerEncoderLayer wrapper that accepts d_ff_factor etc."""

    def __init__(
        self,
        d_model,
        activation,
        d_ff_factor=None,
        dim_feedforward=None,
        norm_first=True,
        **kwargs,
    ):
        if (dim_feedforward is None) == (d_ff_factor is None):
            raise ValueError(
                "Please specify either 'dim_feedforward' "
                f"({dim_feedforward}) XOR 'd_ff_factor' ({d_ff_factor})"
            )
        if dim_feedforward is None:
            dim_feedforward = int(d_model * d_ff_factor)
        activation = ACTIVATION_FUNCTIONS.get(activation, F.relu)
        super().__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            norm_first=norm_first,
            **kwargs,
        )


class SelfAttentionLayer(nn.Module):
    """Transformer-like encoder layer that can return attention weights."""

    def __init__(
        self,
        d_model,
        activation,
        nhead,
        d_ff_factor=None,
        dim_feedforward=None,
        norm_first=True,
        dropout=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        if (dim_feedforward is None) == (d_ff_factor is None):
            raise ValueError(
                "Please specify either 'dim_feedforward' "
                f"({dim_feedforward}) XOR 'd_ff_factor' ({d_ff_factor})"
            )
        if dim_feedforward is None:
            dim_feedforward = int(d_model * d_ff_factor)

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = ACTIVATION_FUNCTIONS.get(activation, F.relu)

    def forward(
        self,
        src: torch.Tensor,
        src_mask=None,
        src_key_padding_mask=None,
        need_weights: bool = False,
    ):
        """Pass the input through the encoder layer.

        Args are consistent with PyTorch's Transformer API.
        """
        x = src
        if self.norm_first:
            x1, weight = self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                need_weights=need_weights,
            )
            x = x + x1
            x = x + self._ff_block(self.norm2(x))
        else:
            x1, weight = self._sa_block(
                x, src_mask, src_key_padding_mask, need_weights=need_weights
            )
            x = self.norm1(x + x1)
            x = self.norm2(x + self._ff_block(x))
        if need_weights:
            return x, weight
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, need_weights: bool = False):
        out, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return self.dropout1(out), weights

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoderOrg(nn.Module):
    """Stack of encoder layers with optional final LayerNorm."""

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, need_weights=False):
        output = src
        weights = []
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                need_weights=need_weights,
            )
            if need_weights:
                output, weight = output
                weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            return output, weights
        return output


# =============================================================================
# Positional encoding / embedding
# =============================================================================
def load_pe_pre_hook_keep(model, state_dict, prefix, local_metadata, strict, missing_keys, upexpected_keys, error_msgs):
    state_dict[prefix + "pe"] = model.pe


def load_pe_pre_hook_load(model, state_dict, prefix, local_metadata, strict, missing_keys, upexpected_keys, error_msgs):
    if prefix + "pe" in state_dict:
        model.register_buffer("pe", state_dict[prefix + "pe"])
    else:
        state_dict[prefix + "pe"] = model.pe


def load_pe_pre_hook_larger(model, state_dict, prefix, local_metadata, strict, missing_keys, upexpected_keys, error_msgs):
    if prefix + "pe" in state_dict and len(model.pe) < len(state_dict[prefix + "pe"]):
        model.register_buffer("pe", state_dict[prefix + "pe"])
    else:
        state_dict[prefix + "pe"] = model.pe


load_pe_pre_hooks = {
    "keep": load_pe_pre_hook_keep,
    "load": load_pe_pre_hook_load,
    "larger": load_pe_pre_hook_larger,
}


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding with dropout (no embedding)."""

    def __init__(self, emb_size: int, dropout: float, max_len: int, load_pe: str = "keep"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)
        self._register_load_state_dict_pre_hook(load_pe_pre_hooks[load_pe], with_module=True)

    def forward(self, input, position: int | None = None):
        """Transpose input to [T, B, F] and add PE.

        Args:
            input: [B, T, F]
            position: Optional single position to pick from pe.

        Returns:
            [T, B, F] after dropout.
        """
        input = input.transpose(0, 1)
        if position is None:
            pe = Variable(self.pe[: input.size(0)], requires_grad=False)
        else:
            pe = Variable(self.pe[position], requires_grad=False)
        return self.dropout(input + pe)


class PositionalEmbedding(nn.Module):
    """Token embedding + sinusoidal positional encoding with optional scaling."""

    def __init__(
        self, embedding: dict, dropout: float, max_len: int, factorize: bool = False, load_pe: str = "keep"
    ):
        super().__init__()
        self.embedding = nn.Embedding(**embedding)
        emb_size = embedding["embedding_dim"]
        self.factorize = factorize
        if self.factorize:
            self.factor = math.sqrt(emb_size)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

        self._register_load_state_dict_pre_hook(load_pe_pre_hooks[load_pe], with_module=True)

    def forward(self, input, position: int | None = None):
        """Embed tokens and add positional encoding, then transpose to [T, B, F]."""
        input = self.embedding(input.transpose(0, 1).contiguous())
        if self.factorize:
            input *= self.factor
        if position is None:
            pe = Variable(self.pe[: input.size(0)], requires_grad=False)
        else:
            pe = Variable(self.pe[position], requires_grad=False)
        return self.dropout(input + pe)


# =============================================================================
# Encoders / Decoders
# =============================================================================
class TransformerEncoder(nn.Module):
    """Transformer encoder built from SelfAttentionLayer blocks."""

    def __init__(self, layer, n_layer, norm=None, init=dict()):
        super().__init__()
        d_model = layer["d_model"]
        layer = SelfAttentionLayer(**layer)
        if norm is not None:
            norm = nn.LayerNorm(normalized_shape=d_model, **norm)
        self.encoder = TransformerEncoderOrg(layer, num_layers=n_layer, norm=norm)

        # Weight init (pattern match by substring)
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name:
                    init_config2func(config)(param)

    def forward(self, src, key_padding_mask, need_weights: bool = False):
        """Forward into encoder.

        Args:
            src: [T, B, D]
            key_padding_mask: [B, T] (bool)

        Returns:
            memory: [T, B, D] (or (memory, weights) if need_weights=True)
        """
        return self.encoder(
            src=src, mask=None, src_key_padding_mask=key_padding_mask, need_weights=need_weights
        )


class TransformerDecoder(nn.Module):
    """(Legacy) Transformer decoder using nn.TransformerDecoderLayer."""

    def __init__(self, layer, n_layer, max_len, norm=None, init=dict()):
        super().__init__()
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer("square_subsequent_mask", square_mask)
        d_model = layer["d_model"]

        # Handle activation semantics
        if "activation" in layer:
            activation = layer["activation"]
            if isinstance(activation, str):
                layer["activation"] = ACTIVATION_FUNCTIONS.get(activation, F.relu)
            else:
                layer["activation"] = function_config2func(activation)
        if "d_ff_factor" in layer:
            layer["dim_feedforward"] = d_model * layer.pop("d_ff_factor")
        layer.setdefault("norm_first", True)

        dec_layer = nn.TransformerDecoderLayer(**layer)
        if norm is not None:
            norm = nn.LayerNorm(normalized_shape=d_model, **norm)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layer, norm=norm)

        # Weight init
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name:
                    init_config2func(config)(param)

    def forward(self, mode: str = "forced", *args, **kwargs):
        """Dispatch to one of ('forced', 'cell_forward', 'prepare_cell_forward')."""
        if mode == "forced":
            return self.forced(*args, **kwargs)
        if mode == "cell_forward":
            return self.cell_forward(*args, **kwargs)
        if mode == "prepare_cell_forward":
            return self.prepare_cell_forward(*args, **kwargs)
        raise ValueError(f"Unsupported type of mode: {mode}")

    def forced(self, tgt, memory, memory_key_padding_mask):
        """Full-sequence decoding with causal mask."""
        length = tgt.shape[0]
        mask = self.square_subsequent_mask[:length, :length]
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        return out

    # NOTE: The following low-level cell decoding logic is kept as-is
    # to preserve behavior, even though it uses private F._scaled_dot_product_attention.

    def cell_forward(self, tgt, mem_attn_mask, ks, vs, mem_ks, mem_vs):
        d_model = tgt.shape[-1]
        x = tgt.squeeze(0)
        for i_layer, layer in enumerate(self.decoder.layers):
            residual = x
            attn = layer.self_attn
            num_heads = attn.num_heads
            bsz, embed_dim = x.shape
            head_dim = embed_dim // num_heads
            q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            q = q.contiguous().view(bsz * num_heads, 1, head_dim)
            k = k.contiguous().view(bsz * num_heads, head_dim).unsqueeze(1)
            v = v.contiguous().view(bsz * num_heads, head_dim).unsqueeze(1)
            ks[i_layer] = torch.cat([ks[i_layer], k], dim=1)
            vs[i_layer] = torch.cat([vs[i_layer], v], dim=1)

            dropout_p = attn.dropout if attn.training else 0.0
            attn_output, _ = F.scaled_dot_product_attention(
                q, ks[i_layer], vs[i_layer], None, dropout_p
            )
            attn_output = attn_output.transpose(0, 1).contiguous().view(bsz, embed_dim)
            attn_output = attn.out_proj(attn_output).view(bsz, -1)
            x = layer.norm1(layer.dropout1(attn_output) + residual)

            residual = x
            attn = layer.multihead_attn
            num_heads = attn.num_heads
            bsz, embed_dim = x.shape
            head_dim = embed_dim // num_heads
            q = F.linear(x, attn.in_proj_weight[: d_model], attn.in_proj_bias[: d_model])
            q = q.contiguous().view(1, bsz * num_heads, head_dim).transpose(0, 1)
            dropout_p = 0.0 if not attn.training else attn.dropout
            attn_output, _ = F.scaled_dot_product_attention(
                q, mem_ks[i_layer], mem_vs[i_layer], mem_attn_mask, dropout_p
            )
            attn_output = attn_output.transpose(0, 1).contiguous().view(bsz, embed_dim)
            attn_output = attn.out_proj(attn_output).view(bsz, -1)
            x = layer.norm2(layer.dropout2(attn_output) + residual)
            x = layer.norm3(
                layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))) + x
            )
        out = self.decoder.norm(x)
        return out.unsqueeze(1)

    def prepare_cell_forward(self, memory, memory_key_padding_mask):
        ilen, bsize, d_model = memory.shape
        nhead = self.decoder.layers[0].multihead_attn.num_heads
        n_layer = len(self.decoder.layers)
        device = memory.device

        mem_attn_mask = torch.zeros_like(memory_key_padding_mask, dtype=memory.dtype)
        mem_attn_mask.masked_fill_(memory_key_padding_mask, float("-inf"))
        mem_attn_mask = (
            mem_attn_mask.view(bsize, 1, 1, ilen).expand(-1, nhead, -1, -1).reshape(bsize * nhead, 1, ilen)
        )

        ks = [
            torch.full((bsize * nhead, 0, d_model // nhead), fill_value=0.0, device=device)
            for _ in range(n_layer)
        ]
        vs = [
            torch.full((bsize * nhead, 0, d_model // nhead), fill_value=0.0, device=device)
            for _ in range(n_layer)
        ]
        mem_ks = []
        mem_vs = []
        for layer in self.decoder.layers:
            attn = layer.multihead_attn
            w_kv = attn.in_proj_weight[d_model:]
            b_kv = attn.in_proj_bias[d_model:]

            kv = F.linear(memory, w_kv, b_kv)
            k, v = kv.chunk(2, dim=-1)
            k = k.contiguous().view(k.shape[0], bsize * nhead, d_model // nhead).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], bsize * nhead, d_model // nhead).transpose(0, 1)
            mem_ks.append(k)
            mem_vs.append(v)
        return mem_attn_mask, ks, vs, mem_ks, mem_vs


# =============================================================================
# Lightweight attention-only decoders
# =============================================================================
def load_square_mask_pre_hook_keep(model, state_dict, prefix, local_metadata, strict, missing_keys, upexpected_keys, error_msgs):
    state_dict[prefix + "square_subsequent_mask"] = model.square_subsequent_mask


def load_square_mask_pre_hook_load(model, state_dict, prefix, local_metadata, strict, missing_keys, upexpected_keys, error_msgs):
    if prefix + "square_subsequent_mask" in state_dict:
        model.register_buffer("square_subsequent_mask", state_dict[prefix + "square_subsequent_mask"])
    else:
        state_dict[prefix + "square_subsequent_mask"] = model.square_subsequent_mask


def load_square_mask_pre_hook_larger(model, state_dict, prefix, local_metadata, strict, missing_keys, upexpected_keys, error_msgs):
    if prefix + "square_subsequent_mask" in state_dict and len(model.square_subsequent_mask) < len(
        state_dict[prefix + "square_subsequent_mask"]
    ):
        model.register_buffer("square_subsequent_mask", state_dict[prefix + "square_subsequent_mask"])
    else:
        state_dict[prefix + "square_subsequent_mask"] = model.square_subsequent_mask


load_square_mask_pre_hooks = {
    "keep": load_square_mask_pre_hook_keep,
    "load": load_square_mask_pre_hook_load,
    "larger": load_square_mask_pre_hook_larger,
}


class LatentSequenceDecoder(nn.Module):
    """Dispatch-only base class to unify decoder APIs."""

    def forward(self, mode: str = "forced", *args, **kwargs):
        method = getattr(self, mode, None)
        if method is not None:
            return method(*args, **kwargs)
        raise ValueError(f"Unsupported type of mode: {mode}")

    def forced(self, *args, **kwargs):
        raise NotImplementedError

    def cell_forward(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_cell_forward(self, *args, **kwargs):
        raise NotImplementedError

    def split_beam(self, *args, **kwargs):
        raise NotImplementedError


class AttentionDecoder(LatentSequenceDecoder):
    """Encoder-only stack used as a decoder over inputs + latent bias."""

    def __init__(self, layer, num_layers, init, max_len, load_square_mask="keep"):
        super().__init__()
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer("square_subsequent_mask", square_mask)
        d_model = layer["d_model"]
        self.d_model = d_model

        decoder_layer = SelfAttentionLayer_old(**layer)
        # Avoid nested tensor optimization warnings; no semantic change.
        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Weight init
        for layer_ in self.decoder.layers:
            for param_name in init:
                init_config2func(init[param_name])(layer_.state_dict()[param_name])

        self._register_load_state_dict_pre_hook(
            load_square_mask_pre_hooks[load_square_mask], with_module=True
        )

    def prepare_cell_forward(self, latent: torch.Tensor):
        """Prepare per-layer cached state for step-wise decoding."""
        batch_size, _ = latent.shape
        return [
            torch.zeros(size=(0, batch_size, self.d_model), dtype=torch.float, device=latent.device)
            for _ in range(self.decoder.num_layers)
        ]

    def gather_beam(self, state, beam_index: torch.Tensor):
        """Reorder cached state by beam index."""
        length, _, d_model = state[0].shape
        batch_size, beam_size = beam_index.shape
        new_state = []
        beam_index = beam_index.view(1, batch_size, beam_size, 1).expand(length, -1, -1, d_model)
        for s in state:
            s = s.view(length, batch_size, beam_size, d_model)
            s = s.gather(dim=2, index=beam_index).view(length, -1, d_model)
            new_state.append(s)
        return new_state

    def forced(self, tgt, latent):
        """Full-sequence pass with latent bias added to all steps."""
        max_len, _, _ = tgt.shape
        tgt = tgt + latent.unsqueeze(0)
        input_mask = self.square_subsequent_mask[:max_len, :max_len]
        output = self.decoder(src=tgt, mask=input_mask, src_key_padding_mask=None)
        return output.transpose(0, 1)

    def cell_forward(self, tgt, latent, state, position):
        """Single-step pass, updating per-layer cached states."""
        cur_output = tgt + latent.unsqueeze(0)
        for i_layer, layer in enumerate(self.decoder.layers):
            prev_y = state[i_layer]
            cur_y = layer.norm1(cur_output)
            y = torch.cat([prev_y, cur_y], dim=0)
            state[i_layer] = y
            cur_attn = layer.self_attn(
                cur_y, y, y, attn_mask=None, key_padding_mask=None, need_weights=False
            )[0]
            cur_output = cur_output + layer.dropout1(cur_attn)
            cur_output = cur_output + layer._ff_block(layer.norm2(cur_output))
        return cur_output.transpose(0, 1), state


class TransformerLMDecoder(LatentSequenceDecoder):
    """Decoder that does not depend on memory/latent (LM-style)."""

    def __init__(self, layer, num_layers, init, max_len, load_square_mask="keep"):
        super().__init__()
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer("square_subsequent_mask", square_mask)
        d_model = layer["d_model"]
        self.d_model = d_model

        decoder_layer = SelfAttentionLayer_old(**layer)
        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Weight init
        for layer_ in self.decoder.layers:
            for param_name in init:
                init_config2func(init[param_name])(layer_.state_dict()[param_name])

        self._register_load_state_dict_pre_hook(
            load_square_mask_pre_hooks[load_square_mask], with_module=True
        )

    def prepare_cell_forward(self, batch_size: int):
        return [
            torch.zeros(
                size=(0, batch_size, self.d_model),
                dtype=torch.float,
                device=self.square_subsequent_mask.device,
            )
            for _ in range(self.decoder.num_layers)
        ]

    def forced(self, tgt):
        max_len, _, _ = tgt.shape
        input_mask = self.square_subsequent_mask[:max_len, :max_len]
        output = self.decoder(src=tgt, mask=input_mask, src_key_padding_mask=None)
        return output.transpose(0, 1)

    def cell_forward(self, tgt, state, position):
        cur_output = tgt
        for i_layer, layer in enumerate(self.decoder.layers):
            prev_y = state[i_layer]
            cur_y = layer.norm1(cur_output)
            y = torch.cat([prev_y, cur_y], dim=0)
            state[i_layer] = y
            cur_attn = layer.self_attn(
                cur_y, y, y, attn_mask=None, key_padding_mask=None, need_weights=False
            )[0]
            cur_output = cur_output + layer.dropout1(cur_attn)
            cur_output = cur_output + layer._ff_block(layer.norm2(cur_output))
        return cur_output.transpose(0, 1), state


# =============================================================================
# Simple classifiers / losses
# =============================================================================
class MLP(nn.Module):
    """Three-layer MLP ending with sigmoid for binary output."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.classifier = nn.Sequential(
            nn.Linear(n_embd, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Flatten inputs to 2D [N, C] and targets to 1D for CE loss."""

    def forward(self, input, target):
        n_class = input.shape[-1]
        return super().forward(
            input=input.contiguous().view(-1, n_class), target=target.ravel()
        )


class BCELoss(nn.BCELoss):
    """Binary cross-entropy with target unsqueezed to [N, 1]."""

    def forward(self, input, target):
        return super().forward(input=input, target=torch.unsqueeze(target.float(), dim=1))


# =============================================================================
# Decoding helpers
# =============================================================================
class GreedyDecoder(nn.Module):
    """Greedy (and beam) decoding utilities.

    Note:
        This class intentionally does not store outputs internally; it operates
        with explicit inputs/outputs to keep side effects predictable.
    """

    def __init__(self, start_token, end_token: int | None = None):
        super().__init__()
        self.start_token = start_token
        if end_token is None:
            print(
                "[WARNING] end_token is not specified in GreedyDecoder.__init__ and "
                "defaulted to 2",
                file=sys.stderr,
            )
            end_token = 2
        self.end_token = end_token
        self._device_param = nn.Parameter(torch.zeros((0,)))

    def forward(self, *args, mode, **kwargs):
        method = getattr(self, mode, None)
        if method is not None:
            return method(*args, **kwargs)
        raise ValueError(f"Unsupported type of mode: {mode}")

    # ----- Greedy -----
    def init(self, batch_size: int):
        cur_input = torch.full(
            (batch_size, 1),
            fill_value=self.start_token,
            dtype=torch.long,
            device=self._device_param.device,
        )
        return cur_input, []

    def add(self, cur_proba, outs):
        """Append argmax token of current step."""
        cur_input = torch.argmax(cur_proba, dim=-1)
        outs.append(cur_input)
        return cur_input, outs

    def sample_add(self, cur_proba, outs):
        """Append multinomial sample from current step distribution."""
        cur_input = torch.multinomial(F.softmax(cur_proba.squeeze(1), dim=-1), num_samples=1)
        outs.append(cur_input)
        return cur_input, outs

    def aggregate(self, outs):
        """Concatenate step-wise tokens to [B, T]."""
        return torch.cat(outs, dim=1)

    # ----- Beam search -----
    def beam_init(self, latent: torch.Tensor, beam_size: int):
        """Initialize beam search tensors."""
        batch_size, latent_size = latent.shape
        device = latent.device
        latent = latent.view(batch_size, 1, latent_size).expand(-1, beam_size, -1).contiguous()
        latent = latent.view(batch_size * beam_size, latent_size)
        cur_input = torch.full(
            (batch_size * beam_size, 1), fill_value=self.start_token, dtype=torch.long, device=device
        )
        is_ended = torch.full((batch_size, beam_size), fill_value=False, dtype=torch.bool, device=device)
        outs = torch.zeros((0, batch_size, beam_size), dtype=torch.long, device=device)
        proba = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        return latent, cur_input, outs, proba, is_ended

    def beam_add(self, cur_proba: torch.Tensor, proba: torch.Tensor, outs: torch.Tensor, is_ended: torch.Tensor):
        """One beam-search step.

        Shapes:
            cur_proba: [B*E, 1, V]
            proba:     [B, E] (logits before softmax)
            outs:      [L, B, E]
            is_ended:  [B, E]
        """
        _, _, voc_size = cur_proba.shape
        length, batch_size, beam_size = outs.shape
        cur_proba = cur_proba.view(batch_size, beam_size, voc_size)

        # Mask finished beams
        cur_proba[is_ended] = -torch.inf
        cur_proba[:, :, self.end_token][is_ended] = 0

        proba = proba.unsqueeze(-1) + cur_proba  # [B, E, V]
        proba = proba.view(batch_size, -1)  # [B, E*V]
        proba, topk_beam_voc = proba.topk(k=beam_size, dim=-1)  # [B, E]
        topk_voc = topk_beam_voc % voc_size  # [B, E]
        topk_beam = torch.div(topk_beam_voc, voc_size, rounding_mode="floor")  # [B, E]

        # Gather previous tokens/states according to new beam indices
        outs = outs.gather(
            dim=-1,
            index=topk_beam.view(1, batch_size, beam_size).expand((length, batch_size, beam_size)),
        )
        is_ended = is_ended.gather(dim=-1, index=topk_beam)

        outs = torch.cat([outs, topk_voc.unsqueeze(0)], dim=0)
        is_ended[topk_voc == self.end_token] = True
        cur_input = topk_voc.view(batch_size * beam_size, 1)
        return cur_input, proba, outs, is_ended, topk_beam

    def beam_aggregate(self, outs: torch.Tensor):
        """Return the best beam (index 0) as [B, L]."""
        return outs[:, :, 0].transpose(0, 1).contiguous()

    # ----- Utility -----
    def force(self, proba, add_start_token: bool = False):
        force = torch.argmax(proba, dim=-1)
        if add_start_token:
            batch_size, _length = force.shape
            force = torch.cat(
                [torch.full((batch_size, 1), fill_value=self.start_token, device=force.device), force],
                dim=1,
            )
        return force


# =============================================================================
# Registry bindings (explicit)
# =============================================================================
register_module("TeacherForcer", TeacherForcer)
register_module("MaskMaker", MaskMaker)
register_module("PositionalEmbedding", PositionalEmbedding)
register_module("TransformerEncoder", TransformerEncoder)
register_module("AttentionDecoder", AttentionDecoder)
register_module("TransformerLMDecoder", TransformerLMDecoder)
register_module("CrossEntropyLoss", CrossEntropyLoss)
register_module("BCELoss", BCELoss)
register_module("GreedyDecoder", GreedyDecoder)
