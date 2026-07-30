"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from x_transformers.x_transformers import RotaryEmbedding

# from src.model.prompt_vp import MRTE
from src.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    AdaLayerNorm_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)

from modules_kvcahe import (
    DiTBlock,
    ChunkDiTBlock,
)

class GlobalTimbreMemory(nn.Module):
    """Global Timbre Memory (GTM)
    Decomposes the global speaker embedding into K reusable timbre prototype slots.
    A speaker-specific 2-layer MLP generates speaker-specific KV, fused with a universal prior.
    """
    def __init__(self, spk_dim=256, memory_slots=8, hidden_dim=256):
        super().__init__()
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim

        # Speaker-specific mapping -- 2-layer MLP for increased expressiveness
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

        # Universal speaker-agnostic prototypes (learns common pronunciation patterns), small-variance init
        self.k_prior = nn.Parameter(torch.zeros(memory_slots, hidden_dim))
        self.v_prior = nn.Parameter(torch.zeros(memory_slots, hidden_dim))
        nn.init.normal_(self.k_prior, std=0.02)
        nn.init.normal_(self.v_prior, std=0.02)

        # LayerNorm after fusion for stable training
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, spks):
        # spks: [B, spk_dim] static global speaker embedding
        B = spks.shape[0]
        # Generate speaker-specific key-value pairs
        k_spk = self.mlp_k(spks).reshape(B, self.memory_slots, self.hidden_dim)
        v_spk = self.mlp_v(spks).reshape(B, self.memory_slots, self.hidden_dim)
        # Fuse with universal prototypes + LayerNorm
        k = self.norm_k(k_spk + torch.tanh(self.k_prior).unsqueeze(0))  # [B, K, D]
        v = self.norm_v(v_spk + torch.tanh(self.v_prior).unsqueeze(0))  # [B, K, D]
        return k, v
    

class TemporalTimbreEncoder(nn.Module):
    """Temporal Timbre Encoder (TVT processing block)
    Frame-level content vectors serve as Query in Multi-Head Cross-Attention with GTM.
    The resulting time-varying timbre features are fused with the global speaker embedding
    via gated Slerp, preserving the unit hypersphere geometry of the speaker embedding.
    """
    def __init__(self, content_dim=256, hidden_dim=256, attn_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        assert attn_dim % num_heads == 0

        # Multi-Head Cross-Attention projections
        self.q_proj = nn.Linear(content_dim, attn_dim)
        self.k_proj = nn.Linear(hidden_dim, attn_dim)   # Input from GTM hidden_dim
        self.v_proj = nn.Linear(hidden_dim, attn_dim)   # Independent V projection
        self.out_proj = nn.Linear(attn_dim, content_dim)    # Output projection back to content_dim

        self.attn_scale = self.head_dim ** -0.5
        self.attn_norm = nn.LayerNorm(content_dim)


    def forward(self, bn, k_mem, v_mem):
        """Args:
            bn:     [B, T, content_dim]  frame-level content features
            k_mem:  [B, K, hidden_dim]   GTM keys
            v_mem:  [B, K, hidden_dim]   GTM values
        Returns:
            timbre_cond: [B, T, content_dim] time-varying timbre conditioning
        """
        B, T, _ = bn.shape
        K = k_mem.shape[1]

        # ---- 1. Multi-Head Cross-Attention: content queries × GTM ----
        q = self.q_proj(bn)     # [B, T, attn_dim]
        k = self.k_proj(k_mem)  # [B, K, attn_dim]
        v = self.v_proj(v_mem)  # [B, K, attn_dim]

        # reshape → [B, num_heads, seq_len, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale  # [B, H, T, K]
        attn = torch.softmax(attn, dim=-1)
        v_t = torch.matmul(attn, v)  # [B, H, T, head_dim]

        # merge heads -> output projection
        v_t = v_t.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, attn_dim]
        timbre_cond = self.attn_norm(self.out_proj(v_t))                # [B, T, content_dim]

        return timbre_cond

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, cond_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + cond_dim * 2, out_dim)
        # self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], spks: float["b n d"], drop_audio_cond=False):  # noqa: F722
    # def forward(self, x: float["b n d"], cond: float["b n d"], timbre_cond: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
            spks = torch.zeros_like(spks)

        x = self.proj(torch.cat((x, cond, spks), dim=-1))
        # x = self.conv_pos_embed(x) + x
        return x



# Transformer backbone using DiT blocks


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
        forward_layers=[0],    # Layer 0 allowed to look ahead
        backward_layers=[0,1,2,3],   # Layer 3 allowed to look behind
        t_f_num=[1,0,0,0], 
        t_p_num=[2,2,1,1],
    ):
        super().__init__()

        self.t_time_embed = TimestepEmbedding(dim)
        self.r_time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, bn_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # GTM + TVT time-varying timbre module (replaces MRTE)
        self.gtm = GlobalTimbreMemory(spk_dim=bn_dim, memory_slots=32, hidden_dim=bn_dim)
        self.temporal_timbre = TemporalTimbreEncoder(
            content_dim=bn_dim, hidden_dim=bn_dim,
            attn_dim=128, num_heads=4,
        )

        forward_layers = set(forward_layers) if forward_layers else set()
        backward_layers = set(backward_layers) if backward_layers else set()

        self.transformer_blocks = nn.ModuleList(
            [
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
                    t_p=t_p_num[i] if i in backward_layers else 0,  # backward
                    t_f=t_f_num[i] if i in forward_layers else 0,    # forward
                )
                for i in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward


    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722           B, T, 80
        t: float["b"] | float[""],  # time step  # noqa: F821 F722
        r: float["b"] | float[""],  # time step  # noqa: F821 F722
        cache: float["b n d"],
        cond: float["b n d"],  # bn  # noqa: F722         B, T, 256
        spks: float["b d"],  # spks  # noqa: F722       B, 256
        offset=0,
        mask: bool["b n"] | None = None,  # noqa: F722
        is_inference: bool = False,
        is_uncondition: bool = False,
        cfg_mask: bool["b"] | None = None,  # noqa: F722
        kv_cache=None,
    ):
        
        batch, seq_len = x.shape[0], x.shape[1]

        # ---- timestep embedding ----
        t = self.t_time_embed(t)
        r = self.r_time_embed(r)
        t = t + r

        # ---- GTM: global speaker embedding -> timbre memory key-value pairs ----
        k_mem, v_mem = self.gtm(spks)  # spks: [B, 256] -> k,v: [B, K, 256]

        # ---- TVT: frame-level BN x GTM -> timbre-enhanced timbre_cond ----
        timbre_cond = self.temporal_timbre(cond, k_mem, v_mem)  # [B, T, bn_dim]

        # Expand spks_global to frame level, as pure global identity condition (independent of timbre_cond)
        spks_expanded = spks.unsqueeze(1).expand(-1, cond.shape[1], -1)  # [B, T, spk_dim]

        # ---- CFG masking ----
        if cfg_mask is not None:
            cfg_mask_ = rearrange(cfg_mask, "b -> b 1 1")
            timbre_cond = torch.where(cfg_mask_, torch.zeros_like(timbre_cond), timbre_cond)
            spks_expanded = torch.where(cfg_mask_, torch.zeros_like(spks_expanded), spks_expanded)

        # Dual-path input: timbre_cond (content+timbre) + spks_expanded (pure global identity)
        x = self.input_embed(x, timbre_cond, spks_expanded, drop_audio_cond=is_uncondition)
        
        # train
        if not is_inference:

            rope = self.rotary_embed.forward_from_seq_len(seq_len)
        # infer
        else:
            if cache != None:
                cache = self.cache_embed(cache)
                x = torch.concat((cache, x), dim=1)   # [b, cache_len + seq_len, dim]
                
                # inference does not need to consider mask
                cache_len = cache.shape[1]
                rope_cache = self.rotary_embed.forward_from_seq_len(cache_len)
                rope_x = self.rotary_embed.forward_from_seq_len(offset + seq_len)
                rope = (torch.concat((rope_cache[0], rope_x[0][:, -seq_len:, :]), dim=1), rope_cache[1])
            else:
                rope = self.rotary_embed.forward_from_seq_len(offset + seq_len)
                

        if self.long_skip_connection is not None:
            residual = x

        new_kv_cache = []
        # inner_hidden_states = []
        for index_block, block in enumerate(self.transformer_blocks):
            if kv_cache is not None:
                block_kv_cache = kv_cache[index_block]
            else:
                block_kv_cache = None
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x, new_block_kv_cache = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, block_kv_cache, use_reentrant=False)
            else:
                x, new_block_kv_cache = block(x, t, mask=mask, rope=rope, is_inference=is_inference, kv_cache=block_kv_cache)
            new_kv_cache.append(new_block_kv_cache)
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        
        
        # x = x[:, -seq_len:, :]
        x = self.norm_out(x, t)
        
        output = self.proj_out(x)
        
        return output, new_kv_cache



