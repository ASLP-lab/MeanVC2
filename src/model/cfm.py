"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from src.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)

from typing import Optional, List, Tuple


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask
        # self.num_channels = 80

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        cache: float["b n d"],
        spks: float["b d"],  # noqa: F722
        # duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        offset=0,
        seed: int | None = None,
        max_duration=10000,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        
        # bn
        cond = cond.to(next(self.parameters()).dtype)
        
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # spks
        spks = spks.unsqueeze(1).repeat(1, cond.shape[1], 1)

        # duration
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask
        duration = cond_seq_len
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        # cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        # cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        # cond_mask = cond_mask.unsqueeze(-1)
        # step_cond = torch.where(
        #     cond_mask, cond, torch.zeros_like(cond)
        # )  # allow direct control (cut cond audio) with lens passed in
        step_cond = cond
        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None 
                    
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x, cache=cache, cond=cond, spks=spks, time=t, mask=mask, drop_audio_cond=False, offset=offset, is_inference=True
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cache=cache, cond=cond, spks=spks, time=t, mask=mask, drop_audio_cond=True, offset=offset, is_inference=True
            )

            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, 80, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        # self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        # out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        # inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        # text: int["b nt"] | list[str],  # noqa: F722
        # *,
        # lens: int["b"] | None = None,  # noqa: F821
        # noise_scheduler: str | None = None,
        mel: torch.Tensor,  # B, T, 80
        bn: torch.Tensor,   # B, T, 256
        spks: torch.Tensor,   # B, 256
        inputs_length: torch.tensor,
           # B, 80, T
    ):
        
        batch, seq_len, _ = mel.size()
        dtype = mel.dtype

        # mel is x1
        x1 = mel # ["b n d"]

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        
        # φ_plus = torch.concat([x1, φ], dim=1)
        
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = bn
        
        original_mask = lens_to_mask(inputs_length, length=seq_len)  # [b, n]
        
        # # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        # if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
        #     drop_audio_cond = True
        #     drop_text = True
        # else:
        #     drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        spks = spks.unsqueeze(1).repeat(1, cond.shape[1], 1)
        pred = self.transformer(
            x=φ, 
            cache=x1,
            cond=cond,
            spks=spks, 
            time=time,
            drop_audio_cond=drop_audio_cond,
            mask=original_mask,
        )
        
        # flow matching loss
        # pred_true = pred[:, seq_len:, :]
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[original_mask]

        return loss.mean(), cond, pred
