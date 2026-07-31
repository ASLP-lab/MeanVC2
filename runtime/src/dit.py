"""
DiT (Diffusion Transformer) for streaming voice conversion.

Architecture:
- GlobalTimbreMemory (GTM): 32-slot timbre memory bank from speaker embedding
- TemporalTimbreEncoder (TVT): frame-level cross-attention of BN features to GTM
- InputEmbedding: fuses noisy mel + timbre_cond + global speaker identity
- ChunkDiTBlock stack: chunk-aware streaming attention with KV-cache
- Flow Matching: single-step ODE (t=1.0 -> 0.0)

Key differences from old MRTE-based model:
- spks is [B, 256] GLOBAL (not per-frame [B, T, 256])
- No `prompts` (prompt mel) parameter needed
- Uses BlockAttnProcessor with t_p/t_f per-layer

"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from x_transformers.x_transformers import RotaryEmbedding

from .dit_modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    precompute_freqs_cis,
    TimestepEmbedding,
)
from .modules import (
    ChunkDiTBlock,
)


# ---------------------------------------------------------------------------
# Global Timbre Memory (GTM)
# ---------------------------------------------------------------------------

class GlobalTimbreMemory(nn.Module):
    """
    Global timbre memory bank.

    Decomposes a speaker embedding into K reusable timbre prototype slots.
    A speaker-specific 2-layer MLP generates speaker KV, fused with
    learnable universal priors via tanh gate + LayerNorm.
    """

    def __init__(self, spk_dim=256, memory_slots=32, hidden_dim=256):
        super().__init__()
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim

        # Speaker-specific mapping — 2-layer MLP
        self.mlp_k = nn.Sequential(
            nn.Linear(spk_dim, spk_dim),
            nn.SiLU(),
            nn.Linear(spk_dim, memory_slots * hidden_dim),
        )
        self.mlp_v = nn.Sequential(
            nn.Linear(spk_dim, spk_dim),
            nn.SiLU(),
            nn.Linear(spk_dim, memory_slots * hidden_dim),
        )

        # Universal priors (small-variance init)
        self.k_prior = nn.Parameter(torch.zeros(memory_slots, hidden_dim))
        self.v_prior = nn.Parameter(torch.zeros(memory_slots, hidden_dim))
        nn.init.normal_(self.k_prior, std=0.02)
        nn.init.normal_(self.v_prior, std=0.02)

        # Post-fusion LayerNorm
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, spks):
        """
        Args:
            spks: [B, spk_dim] static global speaker embedding
        Returns:
            k: [B, K, D] timbre memory keys
            v: [B, K, D] timbre memory values
        """
        B = spks.shape[0]
        k_spk = self.mlp_k(spks).reshape(B, self.memory_slots, self.hidden_dim)
        v_spk = self.mlp_v(spks).reshape(B, self.memory_slots, self.hidden_dim)
        k = self.norm_k(k_spk + torch.tanh(self.k_prior).unsqueeze(0))
        v = self.norm_v(v_spk + torch.tanh(self.v_prior).unsqueeze(0))
        return k, v


# ---------------------------------------------------------------------------
# Temporal Timbre Encoder (TVT)
# ---------------------------------------------------------------------------

class TemporalTimbreEncoder(nn.Module):
    """
    Time-varying timbre encoder.

    Frame-level BN content features act as query, performing multi-head
    cross-attention over GTM keys/values to produce frame-level timbre
    conditioning that varies with phonetic content.
    """

    def __init__(self, content_dim=256, hidden_dim=256, attn_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        assert attn_dim % num_heads == 0

        self.q_proj = nn.Linear(content_dim, attn_dim)
        self.k_proj = nn.Linear(hidden_dim, attn_dim)
        self.v_proj = nn.Linear(hidden_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, content_dim)

        self.attn_scale = self.head_dim ** -0.5
        self.attn_norm = nn.LayerNorm(content_dim)

    def forward(self, bn, k_mem, v_mem):
        """
        Args:
            bn:     [B, T, content_dim] frame-level content features
            k_mem:  [B, K, hidden_dim] GTM keys
            v_mem:  [B, K, hidden_dim] GTM values
        Returns:
            timbre_cond: [B, T, content_dim]
        """
        B, T, _ = bn.shape
        K = k_mem.shape[1]

        q = self.q_proj(bn)
        k = self.k_proj(k_mem)
        v = self.v_proj(v_mem)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn = torch.softmax(attn, dim=-1)
        v_t = torch.matmul(attn, v)

        v_t = v_t.transpose(1, 2).contiguous().view(B, T, -1)
        timbre_cond = self.attn_norm(self.out_proj(v_t))
        return timbre_cond


# ---------------------------------------------------------------------------
# InputEmbedding
# ---------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    """Fuse noisy mel, timbre condition, and global speaker identity."""

    def __init__(self, mel_dim, cond_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + cond_dim * 2, out_dim)

    def forward(self, x, cond, spks, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
            spks = torch.zeros_like(spks)
        x = self.proj(torch.cat((x, cond, spks), dim=-1))
        return x


# ---------------------------------------------------------------------------
# DiT (Diffusion Transformer) — streaming voice conversion
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        bn_dim=256,
        qk_norm=None,
        conv_layers=0,
        chunk_size=8,
        block_size=4,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
        forward_layers=None,
        backward_layers=None,
        t_f_num=None,
        t_p_num=None,
    ):
        super().__init__()

        if forward_layers is None:
            forward_layers = [0]
        if backward_layers is None:
            backward_layers = [0, 1, 2, 3]
        if t_f_num is None:
            t_f_num = [1, 0, 0, 0]
        if t_p_num is None:
            t_p_num = [2, 2, 1, 1]

        self.t_time_embed = TimestepEmbedding(dim)
        self.r_time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, bn_dim, dim)

        # FIXED: added missing cache_embed (was absent in original dit_kvcache.py)
        self.cache_embed = nn.Linear(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # GTM + TVT — replaces old MRTE
        self.gtm = GlobalTimbreMemory(spk_dim=bn_dim, memory_slots=32, hidden_dim=bn_dim)
        self.temporal_timbre = TemporalTimbreEncoder(
            content_dim=bn_dim, hidden_dim=bn_dim,
            attn_dim=128, num_heads=4,
        )

        forward_layers = set(forward_layers or [])
        backward_layers = set(backward_layers or [])

        self.transformer_blocks = nn.ModuleList([
            ChunkDiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                chunk_size=chunk_size,
                block_size=block_size,
                pe_attn_head=pe_attn_head,
                t_p=t_p_num[i] if i in backward_layers else 0,
                t_f=t_f_num[i] if i in forward_layers else 0,
            )
            for i in range(depth)
        ])

        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(
        self,
        x,              # [B, T, mel_dim] noised input
        t,              # [B] start timestep
        r,              # [B] end timestep
        cache=None,     # [B, N, mel_dim] previous mel cache (not used in new streaming)
        cond=None,      # [B, T, bn_dim] BN content features
        spks=None,      # [B, spk_dim] GLOBAL speaker embedding (not per-frame!)
        offset=0,       # int: current position offset for RoPE
        mask=None,      # [B, T] optional padding mask
        is_inference: bool = False,
        is_uncondition: bool = False,
        cfg_mask=None,  # [B] classifier-free guidance mask
        kv_cache=None,  # list of (k, v) tuples per layer
        gtm_kv=None,    # precomputed (k_mem, v_mem) — skip GTM when provided
    ):
        batch, seq_len = x.shape[0], x.shape[1]

        # ---- timestep embedding ----
        t_emb = self.t_time_embed(t)
        r_emb = self.r_time_embed(r)
        t_emb = t_emb + r_emb

        # ---- GTM: speaker → timbre memory KV ----
        if gtm_kv is not None:
            k_mem, v_mem = gtm_kv
        else:
            k_mem, v_mem = self.gtm(spks)  # spks: [B, 256]

        # ---- TVT: frame BN × GTM → time-varying timbre condition ----
        timbre_cond = self.temporal_timbre(cond, k_mem, v_mem)  # [B, T, bn_dim]

        # ---- global speaker identity (per-frame) ----
        spks_expanded = spks.unsqueeze(1).expand(-1, cond.shape[1], -1)  # [B, T, spk_dim]

        # ---- CFG masking ----
        if cfg_mask is not None:
            cfg_mask_ = rearrange(cfg_mask, "b -> b 1 1")
            timbre_cond = torch.where(cfg_mask_, torch.zeros_like(timbre_cond), timbre_cond)
            spks_expanded = torch.where(cfg_mask_, torch.zeros_like(spks_expanded), spks_expanded)

        # ---- dual-path input embedding ----
        x = self.input_embed(x, timbre_cond, spks_expanded, drop_audio_cond=is_uncondition)

        # ---- RoPE ----
        if not is_inference:
            rope = self.rotary_embed.forward_from_seq_len(seq_len)
        else:
            if cache is not None:
                cache_embedded = self.cache_embed(cache)
                x = torch.concat((cache_embedded, x), dim=1)
                cache_len = cache_embedded.shape[1]
                rope_cache = self.rotary_embed.forward_from_seq_len(cache_len)
                rope_x = self.rotary_embed.forward_from_seq_len(offset + seq_len)
                rope = (torch.concat((rope_cache[0], rope_x[0][:, -seq_len:, :]), dim=1), rope_cache[1])
            else:
                rope = self.rotary_embed.forward_from_seq_len(offset + seq_len)

        if self.long_skip_connection is not None:
            residual = x

        new_kv_cache = []
        for idx, block in enumerate(self.transformer_blocks):
            block_kv_cache = kv_cache[idx] if kv_cache is not None else None

            if self.checkpoint_activations:
                x, new_block_kv_cache = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t_emb, mask, rope, block_kv_cache,
                    use_reentrant=False,
                )
            else:
                x, new_block_kv_cache = block(
                    x, t_emb, mask=mask, rope=rope,
                    is_inference=is_inference, kv_cache=block_kv_cache,
                )
            new_kv_cache.append(new_block_kv_cache)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t_emb)
        output = self.proj_out(x)

        return output, new_kv_cache
