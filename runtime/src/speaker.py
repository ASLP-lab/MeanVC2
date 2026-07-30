"""
Speaker embedding extraction using WavLM Large + ECAPA-TDNN.

Loads a fine-tuned speaker verification model and extracts 256-dim speaker
embeddings from reference audio for use in the VC pipeline.

Source: meanvc_run/speaker_verification/ (ecapa_tdnn.py + verification.py)

Dependencies: torch, torchaudio, soundfile, numpy
Optional: s3prl (for WavLM feature extraction; install from source if needed)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
from torchaudio.transforms import Resample


# ---------------------------------------------------------------------------
# Lightweight ECAPA-TDNN components (self-contained, no s3prl for fbank mode)
# ---------------------------------------------------------------------------

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width) for _ in range(self.nums)
        ])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = None
        for i in range(self.nums):
            sp = spx[i] if sp is None else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, scale, se_bottleneck_dim):
        super().__init__()
        # Submodule names must match the reference models/ecapa_tdnn.py
        # so that the fine-tuned checkpoint weights load correctly.
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x_in))), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


# ---------------------------------------------------------------------------
# ECAPA-TDNN with WavLM feature extractor
# ---------------------------------------------------------------------------

class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN speaker embedding model with WavLM Large backbone."""

    def __init__(self, feat_dim=1024, channels=512, emb_dim=256,
                 feat_type='wavlm_large', sr=16000,
                 feature_selection="hidden_states", update_extract=False,
                 config_path=None):
        super().__init__()

        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        if feat_type == "fbank" or feat_type == "mfcc":
            self.update_extract = False
            win_len = int(sr * 0.025)
            hop_len = int(sr * 0.01)
            if feat_type == 'fbank':
                self.feature_extract = trans.MelSpectrogram(
                    sample_rate=sr, n_fft=512, win_length=win_len,
                    hop_length=hop_len, f_min=0.0, f_max=sr // 2,
                    pad=0, n_mels=feat_dim,
                )
            else:
                melkwargs = {'n_fft': 512, 'win_length': win_len, 'hop_length': hop_len,
                             'f_min': 0.0, 'f_max': sr // 2, 'pad': 0}
                self.feature_extract = trans.MFCC(
                    sample_rate=sr, n_mfcc=feat_dim, log_mels=False, melkwargs=melkwargs,
                )
        else:
            if config_path is not None:
                # Build UpstreamExpert from tiny config (~10 KB), skipping the
                # 1.2 GB wavlm_large.pt.  Replicates UpstreamExpert.__init__
                # (expert.py:34-54) but without torch.load(ckpt) and
                # load_state_dict — the fine-tuned ckpt provides all weights.
                from s3prl.upstream.wavlm.WavLM import WavLM, WavLMConfig
                from s3prl.upstream.wavlm.expert import UpstreamExpert
                from s3prl.upstream.interfaces import UpstreamBase

                cfg_dict = torch.load(config_path, map_location='cpu')
                cfg = WavLMConfig(cfg_dict)
                wavlm = WavLM(cfg)
                wavlm.feature_grad_mult = 0.0
                wavlm.encoder.layerdrop = 0.0

                expert = UpstreamExpert.__new__(UpstreamExpert)
                UpstreamBase.__init__(expert)
                expert.cfg = cfg
                expert.model = wavlm
                expert.model.feature_grad_mult = 0.0
                expert.model.encoder.layerdrop = 0.0

                if len(expert.hooks) == 0:
                    for module_id in range(len(wavlm.encoder.layers)):
                        expert.add_hook(
                            f"self.model.encoder.layers[{module_id}]",
                            lambda input, output: input[0].transpose(0, 1),
                        )
                    expert.add_hook("self.model.encoder",
                                    lambda input, output: output[0])
                expert._init_layerdrop = wavlm.encoder.layerdrop

                self.feature_extract = expert
            else:
                from s3prl.upstream.wavlm.expert import UpstreamExpert
                self.feature_extract = UpstreamExpert(
                    ckpt='models/wavlm_large.pt'  # fallback, normally not used
                )

            # Disable fp32_attention for layers that have it (compatibility)
            if len(self.feature_extract.model.encoder.layers) == 24:
                for layer_idx in [11, 23]:
                    layer = self.feature_extract.model.encoder.layers[layer_idx]
                    if hasattr(layer.self_attn, "fp32_attention"):
                        layer.self_attn.fp32_attention = False

            self.feat_num = self._get_feat_num()
            self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type != 'fbank' and feat_type != 'mfcc':
            freeze_list = ['final_proj', 'label_embs_concat', 'mask_emb', 'project_q', 'quantizer']
            for name, param in self.feature_extract.named_parameters():
                for freeze_val in freeze_list:
                    if freeze_val in name:
                        param.requires_grad = False
                        break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1],
                                   kernel_size=3, stride=1, padding=2, dilation=2,
                                   scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2],
                                   kernel_size=3, stride=1, padding=3, dilation=3,
                                   scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3],
                                   kernel_size=3, stride=1, padding=4, dilation=4,
                                   scale=8, se_bottleneck_dim=128)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128,
                                          global_context_att=False)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def _get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        return 1

    def _get_feat(self, x):
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                if self.feat_type == 'fbank' or self.feat_type == 'mfcc':
                    x = self.feature_extract(x) + 1e-6
                else:
                    x = self.feature_extract([sample for sample in x])

        if self.feat_type == 'fbank':
            x = x.log()

        if self.feat_type != "fbank" and self.feat_type != "mfcc":
            x = x[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)
            else:
                x = x.unsqueeze(0)
            norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self._get_feat(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)
        return out


def ECAPA_TDNN_SMALL(feat_dim, emb_dim=256, feat_type='fbank', sr=16000,
                     feature_selection="hidden_states", update_extract=False,
                     config_path=None):
    return ECAPA_TDNN(
        feat_dim=feat_dim, channels=512, emb_dim=emb_dim,
        feat_type=feat_type, sr=sr,
        feature_selection=feature_selection,
        update_extract=update_extract,
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_speaker_model(ckpt_path=None, device='cpu', wavlm_config=None):
    """
    Load the WavLM Large + ECAPA-TDNN speaker verification model.

    Args:
        ckpt_path: path to fine-tuned checkpoint (wavlm_large_finetune.pth).
            Contains ALL backbone + ECAPA-TDNN weights.
        device: 'cpu' or 'cuda'
        wavlm_config: path to WavLM config (wavlm_large_cfg.pt, ~10 KB).
            Extracted from wavlm_large.pt via extract_wavlm_config.py.
            When provided, skips loading the 1.2 GB base checkpoint — the
            WavLM backbone is built from config and all weights come from
            ckpt_path via load_state_dict().
    Returns:
        model: ECAPA_TDNN model in eval mode, on the specified device
    """
    model = ECAPA_TDNN_SMALL(
        feat_dim=1024, emb_dim=256,
        feat_type='wavlm_large',
        feature_selection="hidden_states",
        update_extract=False,
        config_path=wavlm_config,
    )
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def extract_embedding(model, wav, sample_rate=16000, device='cpu'):
    """
    Extract 256-dim speaker embedding from audio.

    Args:
        model: ECAPA_TDNN model from init_speaker_model()
        wav: either a file path (str) or a torch.Tensor [1, samples] or numpy array
        sample_rate: target sample rate (used if loading from file)
        device: compute device

    Returns:
        torch.Tensor [1, 256] — L2-normalized speaker embedding
    """
    import soundfile as sf

    if isinstance(wav, str):
        data, sr = sf.read(wav)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        wav = torch.from_numpy(data).unsqueeze(0).float().to(device)
        if sr != sample_rate:
            resample = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
            wav = resample(wav)
    elif isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)
    elif isinstance(wav, torch.Tensor):
        wav = wav.to(device)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

    with torch.no_grad():
        emb = model(wav)

    # NOTE: No L2 normalization — aligned with extract_spk_emb_wavlm_multi_mp3.py
    # which outputs raw (unnormalized) embeddings.
    return emb
