"""
Core neural network modules for MeanVC streaming DiT.

Key components:
- Attention + BlockAttnProcessor (chunk-aware streaming attention with KV-cache)
- DiTBlock, ChunkDiTBlock (transformer blocks with AdaLayerNorm modulation)
- scaled_dot_product_attention_only (custom SDPA with mask support)

"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb

from .dit_modules import (
    AdaLayerNorm,
    AdaLayerNorm_Final,
    FeedForward,
    RMSNorm,
)


# ---------------------------------------------------------------------------
# Attention class
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(
        self,
        processor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0+")

        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if self.context_dim is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_dim is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x,
        c=None,
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,
        c_rope=None,
        is_inference=False,
        kv_cache=None,
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope,
                                  is_inference=is_inference, kv_cache=kv_cache)
        else:
            return self.processor(self, x, mask=mask, rope=rope,
                                  is_inference=is_inference, kv_cache=kv_cache)


# ---------------------------------------------------------------------------
# AttnProcessor (standard, no KV-cache)
# ---------------------------------------------------------------------------

class AttnProcessor:
    def __init__(self, pe_attn_head: int | None = None):
        self.pe_attn_head = pe_attn_head

    def __call__(
        self,
        attn: Attention,
        x,
        mask=None,
        rope=None,
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x


# ---------------------------------------------------------------------------
# Custom SDPA (without flash attention, supports manual mask + GQA)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention_only(query, key, value, attn_mask=None, dropout_p=0.0,
                                       is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    B = query.size(0)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(B, 1, L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(B, 1, L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask[:, :, -attn_bias.shape[2]:, :].logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


# ---------------------------------------------------------------------------
# BlockAttnProcessor — chunk+block-level streaming attention with KV-cache
# ---------------------------------------------------------------------------

class BlockAttnProcessor:
    """
    Chunk-and-block aware attention processor for streaming inference.

    Each chunk is divided into blocks. The attention mask allows:
    - Same chunk: full visibility
    - Past chunks: up to `t_p` previous chunks
    - Future chunk: first `t_f` blocks of the next chunk

    Inference mode uses left-looking causal mask only (no future).
    KV-cache stores all past keys/values; new KV is trimmed by block_size.
    """

    def __init__(self, chunk_size: int, block_size: int, t_p: int, t_f: int,
                 pe_attn_head: int | None = None):
        self.pe_attn_head = pe_attn_head
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.t_p = t_p
        self.t_f = t_f

    def try_cached_mask(self, seq_len, device):
        """Build chunk+block attention mask [seq_len, seq_len]."""
        idx = torch.arange(seq_len, device=device)
        ci = idx // self.chunk_size          # chunk index per position

        qi = ci[:, None]                     # query chunk  [L, 1]
        kj = ci[None, :]                     # key chunk    [1, L]

        # same chunk
        same_chunk = (qi == kj)

        # previous t_p chunks
        prev_chunk = (kj >= (qi - self.t_p)) & (kj < qi)

        # first t_f blocks of the next chunk
        offset_in_chunk = (idx % self.chunk_size)[None, :]
        next_chunk_first_block = (kj == (qi + 1)) & (offset_in_chunk < self.block_size * self.t_f)

        computed_mask = same_chunk | prev_chunk | next_chunk_first_block
        return computed_mask

    def __call__(
        self,
        attn: Attention,
        x,
        mask=None,
        rope=None,
        is_inference=False,
        kv_cache=None,
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]
        device = x.device

        # import pdb; pdb.set_trace()

        # 1. QKV projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # 2. Reshape to multi-head
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 3. QK norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # 4. KV cache
        if kv_cache is None:
            key_cache, value_cache = None, None
        else:
            key_cache, value_cache = kv_cache

        if key_cache is not None:
            key = torch.cat([key_cache, key], dim=2)
        if value_cache is not None:
            value = torch.cat([value_cache, value], dim=2)

        # Trim KV cache: remove future block to prevent leakage
        new_kv_cache = (
            key[:, :, :-self.block_size, :],
            value[:, :, :-self.block_size, :],
        )

        # Sliding window: keep only the last N frames needed by this layer's
        # attention pattern (t_p past chunks + current chunk).  Unbounded
        # growth would OOM on long-running streams.
        max_cache_frames = (self.t_p + 1) * self.chunk_size
        if new_kv_cache[0].shape[2] > max_cache_frames:
            new_kv_cache = (
                new_kv_cache[0][:, :, -max_cache_frames:, :],
                new_kv_cache[1][:, :, -max_cache_frames:, :],
            )

        batch_size, _, seq_len, _ = key.shape

        # 5. Rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # 6. Build attention mask
        computed_mask = self.try_cached_mask(seq_len, device).unsqueeze(0).expand(batch_size, -1, -1)
        attn_mask = computed_mask.unsqueeze(1).expand(batch_size, 1, seq_len, seq_len)

        # 7. Scaled dot-product attention
        attn_output = scaled_dot_product_attention_only(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # 8. Merge heads and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        attn_output = attn_output.to(query.dtype)

        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)

        if mask is not None:
            attn_output = attn_output.masked_fill(~mask.unsqueeze(-1), 0.0)

        return attn_output, new_kv_cache


# ---------------------------------------------------------------------------
# DiTBlock (standard, no KV-cache)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, qk_norm=None,
                 pe_attn_head=None):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(pe_attn_head=pe_attn_head),
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, qk_norm=qk_norm,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x


# ---------------------------------------------------------------------------
# ChunkDiTBlock (streaming, with BlockAttnProcessor + KV-cache)
# ---------------------------------------------------------------------------

class ChunkDiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, qk_norm=None,
                 chunk_size=16, block_size=8, t_p=0, t_f=0, pe_attn_head=None):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=BlockAttnProcessor(
                chunk_size=chunk_size, block_size=block_size,
                t_p=t_p, t_f=t_f, pe_attn_head=pe_attn_head,
            ),
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, qk_norm=qk_norm,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None, is_inference=False, kv_cache=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output, new_kv_cache = self.attn(
            x=norm, mask=mask, rope=rope, is_inference=is_inference, kv_cache=kv_cache
        )
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x, new_kv_cache
