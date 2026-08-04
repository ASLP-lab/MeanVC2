#!/usr/bin/env python3
"""
MeanVC Real-Time Streaming Voice Conversion — terminal inference.

Streaming pipeline:
  Microphone → ASR (FastU2++ JIT) → BN features → DiT (GTM+TVT) → Vocos → Speaker

Supports two model configurations:
  --model 120ms  (chunk_size=12, block_size=4)  → 120ms+40ms
  --model 40ms   (chunk_size=4,  block_size=4)  → 40ms+40ms

Usage:
  # File mode
  python run_rt.py --mode file --input in.wav --output out.wav --model 120ms [--device cuda]

  # Realtime mode
  python run_rt.py --mode realtime --model 40ms [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.dit import DiT
from src.speaker import init_speaker_model, extract_embedding

# ---------------------------------------------------------------------------
# Model paths — relative to meanvc2 root
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCODER_PATH = os.path.join(_ROOT, "ckpts/vocos/vocos.pt")
SPEAKER_MODEL_PATH = os.path.join(_ROOT, "preprocess/ckpts/wavlm_large_finetune.pth")
WAVLM_CONFIG_PATH = os.path.join(_ROOT, "preprocess/ckpts/wavlm_large_cfg.pt")

# Per-model paths + ASR parameters
#   80ms ASR (40+40 VC):  fastu2pp_80ms.pt,  BN_WINDOW=11, BN_STRIDE=8
#   160ms ASR (120+40 VC): fastu2pp_160ms.pt, BN_WINDOW=19, BN_STRIDE=16
MODEL_PATHS = {
    "120ms": {
        "ckpt": os.path.join(_ROOT, "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
        "config": os.path.join(_ROOT, "src/config/config_120ms_40ms.json"),
        "asr_ckpt": os.path.join(_ROOT, "preprocess/ckpts/fastu2pp_160ms.pt"),
        "bn_window": 19,
        "bn_stride": 16,
        "required_cache_size": 8,
        "asr_offset_init": 8,
        "asr_offset_step": 4,
    },
    "40ms": {
        "ckpt": os.path.join(_ROOT, "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors"),
        "config": os.path.join(_ROOT, "src/config/config_40ms_40ms.json"),
        "asr_ckpt": os.path.join(_ROOT, "preprocess/ckpts/fastu2pp_80ms.pt"),
        "bn_window": 11,
        "bn_stride": 8,
        "required_cache_size": 4,
        "asr_offset_init": 4,
        "asr_offset_step": 2,
    },
}

# Default target speaker reference
DEFAULT_TARGET_WAV = "test_audio/common_voice_en_10119832.wav"


# ---------------------------------------------------------------------------
# VC Runner
# ---------------------------------------------------------------------------

class VCRunner:
    """Streaming voice conversion runner."""

    def __init__(self, target_wav: str, device: str = "cpu", model: str = "120ms"):
        self.device = device
        torch.set_num_threads(1)

        paths = MODEL_PATHS[model]
        vc_ckpt = paths["ckpt"]
        vc_config = paths["config"]
        self._asr_ckpt = paths["asr_ckpt"]
        self._bn_window = paths["bn_window"]
        self._bn_stride = paths["bn_stride"]
        self._required_cache_size = paths["required_cache_size"]
        self._asr_offset_init = paths["asr_offset_init"]
        self._asr_offset_step = paths["asr_offset_step"]

        # --- ASR (JIT Fast-U2++) ---
        print("[Init] Loading ASR encoder (JIT)...")
        self.asr = torch.jit.load(self._asr_ckpt)
        self.asr.eval()

        # --- VC ---
        print("[Init] Loading VC model...")
        self.vc = _load_vc_model(vc_config, vc_ckpt, device)

        # --- Vocoder ---
        print("[Init] Loading vocoder...")
        self.vocoder = torch.jit.load(VOCODER_PATH)
        if device == "cuda":
            self.vocoder = self.vocoder.to(device)

        # --- Speaker ---
        print("[Init] Loading speaker model...")
        self.spk_model = init_speaker_model(SPEAKER_MODEL_PATH, device,
                                            wavlm_config=WAVLM_CONFIG_PATH)

        print(f"[Init] Extracting target speaker embedding from: {target_wav}")
        self.vc_spk_emb = extract_embedding(self.spk_model, target_wav, device=device)
        print(f"[Init] Speaker embedding shape: {self.vc_spk_emb.shape}")

        # Precompute GTM KV — speaker is fixed for the entire session
        with torch.no_grad():
            self.vc_gtm_kv = self.vc.gtm(self.vc_spk_emb)
      
        # --- Streaming parameters ---
        self.chunk_size = 12 if model == "120ms" else 4
        self.block_size = 4
        self.CHUNK = 2560  # samples per audio chunk (160ms at 16kHz)

        # --- Vocoder overlap-add ---
        self.vocoder_overlap = 2
        self.upsample_factor = 160
        self.vocoder_wav_overlap = self.upsample_factor  # 160
        self.vocoder_wav_pad = (self.vocoder_overlap - 1) * self.upsample_factor - self.vocoder_wav_overlap
        self.down_linspace = torch.linspace(1, 0, steps=self.vocoder_wav_overlap)
        self.up_linspace = torch.linspace(0, 1, steps=self.vocoder_wav_overlap)

        self._init_cache()

    def _init_cache(self):
        self.samples_cache_len = 400 + 2 * 160
        self.samples_cache = np.full(self.samples_cache_len, -0.5, dtype=np.float32)
        self.fbank_cache = None
        self.encoder_output_cache = None
        self.asr_offset = self._asr_offset_init
        self.asr_att_cache = torch.zeros(6, 4, self._required_cache_size, 128)
        self.asr_cnn_cache = torch.zeros(6, 1, 256, 8)

        self.vc_kv_cache = None
        self.vc_offset = 0
        self.noise_cache = None

        self.vocoder_cache = None
        self.last_wav = None

        self.bn_buffer = None
        self._min_bn_len = self.chunk_size + self.block_size  # 16 or 8

    # ------------------------------------------------------------------
    # ASR encoding (JIT Fast-U2++, adapted from original WeNet version)
    # ------------------------------------------------------------------

    def _encode_chunk(self, samples: np.ndarray) -> torch.Tensor | None:
        with torch.no_grad():
            padded = np.concatenate((self.samples_cache, samples))
            self.samples_cache = padded[-self.samples_cache_len:]

            # Extract fbanks (same as original)
            wav_scaled = padded * (1 << 15)
            wav_t = torch.from_numpy(wav_scaled).unsqueeze(0)
            fbanks = kaldi.fbank(wav_t, frame_length=25, frame_shift=10,
                                 snip_edges=True, num_mel_bins=80,
                                 energy_floor=0.0, dither=0.0,
                                 sample_frequency=16000)

            # Prepend cached fbank frames for continuity
            had_fbank_cache = self.fbank_cache is not None
            if had_fbank_cache:
                fbanks = torch.cat([self.fbank_cache, fbanks], dim=0)
            self.fbank_cache = fbanks[-3:]  # cache overlap for next chunk

            # Reset ASR offset periodically — preserve cache-relative position
            if self.asr_offset >= 4000:
                cache_t = self.asr_att_cache.size(2) if self.asr_att_cache.dim() > 0 else 0
                self.asr_offset = max(cache_t, self.asr_offset - 4000)

            # JIT ASR: sliding window of fbank frames → BN frames per call
            bns = []
            i = 0
            while i + self._bn_window <= fbanks.shape[0]:
                fb = fbanks[i:i+self._bn_window].unsqueeze(0)
                out, self.asr_att_cache, self.asr_cnn_cache = self.asr(
                    fb,
                    torch.tensor(self.asr_offset, dtype=torch.int64),
                    torch.tensor(self._required_cache_size, dtype=torch.int64),
                    self.asr_att_cache, self.asr_cnn_cache,
                )
                bns.append(out.squeeze(0).detach())
                self.asr_offset += self._asr_offset_step
                i += self._bn_stride

            if not bns:
                return None

            # Build BN output (same as original WeNet output format)
            bn = torch.cat(bns, dim=0)  # [N, 256]
            bn = bn.unsqueeze(0)        # [1, N, 256]

            # Drop first 2 BN from fbank cache overlap, cache last frame
            if had_fbank_cache and len(bns) > 1:
                bn = bn[:, 2:, :]

            if self.encoder_output_cache is not None:
                bn = torch.cat([self.encoder_output_cache, bn], dim=1)
            self.encoder_output_cache = bn[:, -1:, :]

            # 4x interpolate → vc_chunk+1 frames, drop first (same as original)
            vc_chunk = 16
            if bn.shape[1] >= 2:
                bn_up = bn.transpose(1, 2)
                bn_up = F.interpolate(bn_up, size=vc_chunk + 1,
                                      mode='linear', align_corners=True)
                bn_up = bn_up.transpose(1, 2)
                bn_up = bn_up[:, 1:, :]  # [1, vc_chunk, 256]
            else:
                bn_up = None
        return bn_up

    # ------------------------------------------------------------------
    # VC step (N-step flow-matching ODE)
    # ------------------------------------------------------------------

    def _vc_step(self, cond: torch.Tensor) -> torch.Tensor:
        N = 2  # ODE steps
        dt = 1.0 / N

        with torch.no_grad():
            x = torch.randn(1, cond.shape[1], 80, device=cond.device, dtype=cond.dtype)

            if self.noise_cache is not None:
                x[:, :self.block_size, :] = self.noise_cache
            self.noise_cache = x[:, -self.block_size:, :].clone()

            if self.vc_offset >= 4000:
                self.vc_offset = 0
                self.vc_kv_cache = None

            pre_kv_cache = self.vc_kv_cache

            for i in range(N):
                t_val = 1.0 - i * dt
                r_val = max(0.0, t_val - dt)

                t_tensor = torch.full((1,), t_val, device=cond.device)
                r_tensor = torch.full((1,), r_val, device=cond.device)

                u, tmp_kv_cache = self.vc(
                    x, t_tensor, r_tensor,
                    cache=None, cond=cond, spks=self.vc_spk_emb,
                    offset=self.vc_offset, is_inference=True,
                    kv_cache=pre_kv_cache, gtm_kv=self.vc_gtm_kv,
                )
                x = x - dt * u

                if i == N - 1:
                    self.vc_kv_cache = tmp_kv_cache

            self.vc_offset += self.chunk_size

            mel = x[:, :self.chunk_size, :].transpose(1, 2)  # [1, 80, chunk_size]
        return mel

    # ------------------------------------------------------------------
    # Vocoder
    # ------------------------------------------------------------------

    def _decode_mel(self, mel: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            if self.vocoder_cache is not None:
                mel = torch.cat([self.vocoder_cache, mel], dim=-1)
            self.vocoder_cache = mel[:, :, -self.vocoder_overlap:]

            mel = (mel + 1) / 2
            if self.device == "cuda":
                mel = mel.to(self.device)
            wav = self.vocoder.decode(mel).squeeze().detach().cpu().numpy()

            if self.last_wav is not None:
                wav = wav[self.vocoder_wav_pad:]
                front = wav[:self.vocoder_wav_overlap]
                smooth_front = (
                    self.last_wav * self.down_linspace.numpy()
                    + front * self.up_linspace.numpy())
                new_wav = np.concatenate([
                    smooth_front,
                    wav[self.vocoder_wav_overlap:-self.vocoder_wav_overlap],
                ])
            else:
                new_wav = wav[:-self.vocoder_wav_overlap]

            self.last_wav = wav[-self.vocoder_wav_overlap:]
        return new_wav

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def process_chunk(self, samples: np.ndarray) -> np.ndarray | None:
        bn_new = self._encode_chunk(samples)
        if bn_new is None:
            return None
        if self.device == "cuda":
            bn_new = bn_new.to(self.device)

        if self.bn_buffer is None:
            self.bn_buffer = bn_new
        else:
            self.bn_buffer = torch.cat([self.bn_buffer, bn_new], dim=1)

        if self.bn_buffer.shape[1] < self._min_bn_len:
            return None

        wav_parts = []
        max_steps = 4
        for _ in range(max_steps):
            if self.bn_buffer.shape[1] < self._min_bn_len:
                break
            cond = self.bn_buffer[:, :self._min_bn_len, :]
            self.bn_buffer = self.bn_buffer[:, self.chunk_size:, :]

            mel = self._vc_step(cond)
            wav = self._decode_mel(mel)
            wav_parts.append(wav)

        if not wav_parts:
            return None
        return np.concatenate(wav_parts)

    # ------------------------------------------------------------------
    # File mode
    # ------------------------------------------------------------------

    def process_file(self, input_path: str, output_path: str, seed: int = 42):
        import soundfile as sf

        wav, sr = sf.read(input_path)
        if wav.ndim == 2:
            wav = np.mean(wav, axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        wav = wav.astype(np.float32)

        self._init_cache()
        torch.manual_seed(seed)

        output_parts = []
        pos = 0
        print(f"[File] Processing {len(wav) / sr:.1f}s audio...")
        t_start = time.time()

        while pos < len(wav):
            end = min(pos + self.CHUNK, len(wav))
            chunk = wav[pos:end]  # no zero-padding — avoids silence artifacts at tail
            pos = end

            out = self.process_chunk(chunk.astype(np.float32))
            if out is not None:
                output_parts.append(out)

        # Flush remaining BN buffer — replicate last BN frame as future context
        while self.bn_buffer is not None and self.bn_buffer.shape[1] >= self.chunk_size:
            pad_cur = self.bn_buffer[:, :self.chunk_size, :]
            if self.bn_buffer.shape[1] > self.chunk_size:
                pad_fut = self.bn_buffer[:, self.chunk_size:self._min_bn_len, :]
            else:
                last_frame = self.bn_buffer[:, -1:, :]
                pad_fut = last_frame.repeat(1, self.block_size, 1)
            cond = torch.cat([pad_cur, pad_fut], dim=1)
            self.bn_buffer = self.bn_buffer[:, self.chunk_size:, :]
            if self.bn_buffer.shape[1] == 0:
                self.bn_buffer = None
            mel = self._vc_step(cond)
            output_parts.append(self._decode_mel(mel))

        # Drain trailing partial BN — pad with edge replication
        if self.bn_buffer is not None and self.bn_buffer.shape[1] > 0:
            n_rem = self.bn_buffer.shape[1]
            last_frame = self.bn_buffer[:, -1:, :]
            pad_cur = last_frame.repeat(1, self.chunk_size, 1)
            pad_cur[:, :n_rem, :] = self.bn_buffer
            pad_fut = last_frame.repeat(1, self.block_size, 1)
            cond = torch.cat([pad_cur, pad_fut], dim=1)
            self.bn_buffer = None
            mel = self._vc_step(cond)
            output_parts.append(self._decode_mel(mel))

        if self.last_wav is not None:
            output_parts.append(self.last_wav)

        final_wav = np.concatenate(output_parts)
        sf.write(output_path, final_wav, 16000)
        elapsed = time.time() - t_start
        rtf = elapsed / (len(wav) / 16000)
        print(f"[File] Done. Output: {output_path}")
        print(f"[File] Duration: {len(wav)/16000:.1f}s, Elapsed: {elapsed:.1f}s, RTF: {rtf:.3f}")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_vc_model(config_path: str, ckpt_path: str, device: str = "cpu") -> DiT:
    with open(config_path) as f:
        model_config = json.load(f)
    model = DiT(**model_config["model"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[VC] Total parameters: {total_params:,}")
    model = model.to(device)

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        weights = load_file(ckpt_path)
        model.load_state_dict(weights, strict=False)
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location=device)
        if "ema_model_state_dict" in checkpoint:
            weights = {k.replace("ema_model.", ""): v
                       for k, v in checkpoint["ema_model_state_dict"].items()
                       if k.replace("ema_model.", "") not in ("initted", "step")}
        elif "model_state_dict" in checkpoint:
            weights = checkpoint["model_state_dict"]
        else:
            weights = checkpoint
        model.load_state_dict(weights, strict=False)

    model.float().eval()
    return model


# ---------------------------------------------------------------------------
# Realtime mode
# ---------------------------------------------------------------------------

def run_realtime(vc: VCRunner, in_device: int | None = None, out_device: int | None = None):
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: Install sounddevice (`pip install sounddevice`)")
        sys.exit(1)

    vc._init_cache()
    chunk_count = 0
    last_print_chunk = -1
    out_buffer = np.array([], dtype=np.float32)

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal chunk_count, last_print_chunk, out_buffer
        if status:
            print(f"[Audio] Status: {status}")

        t0 = time.perf_counter()
        samples = indata[:, 0].astype(np.float32)
        out = vc.process_chunk(samples)
        t1 = time.perf_counter()

        if out is not None:
            out_buffer = np.concatenate([out_buffer, out])

        n_needed = len(outdata)
        if len(out_buffer) >= n_needed:
            outdata[:, 0] = out_buffer[:n_needed]
            out_buffer = out_buffer[n_needed:]
        else:
            n_avail = len(out_buffer)
            outdata[:n_avail, 0] = out_buffer
            outdata[n_avail:, 0] = 0.0
            out_buffer = np.array([], dtype=np.float32)

        chunk_count += 1
        proc_ms = (t1 - t0) * 1000
        chunk_ms = len(samples) / 16
        buf_ms = len(out_buffer) / 16
        if chunk_count - last_print_chunk >= 10 or proc_ms > chunk_ms:
            flag = "⚠ OVERRUN" if proc_ms > chunk_ms else "OK"
            print(f"[{chunk_count}] {proc_ms:.1f}ms / {chunk_ms:.0f}ms {flag}  buf={buf_ms:.0f}ms")
            last_print_chunk = chunk_count

    print("\n[Stream] Using sounddevice for real-time I/O")
    sample_rate = 16000
    blocksize = vc.CHUNK

    print("[Stream] Warming up...")
    warmup = np.zeros(blocksize, dtype=np.float32)
    for _ in range(3):
        vc.process_chunk(warmup)
    vc._init_cache()
    print("[Stream] Starting... (Ctrl+C to stop)")

    try:
        with sd.Stream(
            samplerate=sample_rate, blocksize=blocksize,
            device=(in_device, out_device),
            channels=1, callback=audio_callback, dtype='float32',
        ):
            print("[Stream] Running. Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n[Stream] Stopped by user.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MeanVC Real-Time Voice Conversion")
    parser.add_argument("--target-spk", type=str, default=DEFAULT_TARGET_WAV)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model", type=str, default="120ms", choices=["120ms", "40ms"])
    parser.add_argument("--mode", type=str, default="realtime", choices=["realtime", "file"])
    parser.add_argument("--input", type=str, default=None, help="Input WAV (file mode)")
    parser.add_argument("--output", type=str, default="output_vc.wav", help="Output WAV (file mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (file mode)")
    args = parser.parse_args()

    print("=" * 50)
    print("MeanVC Real-Time Streaming Voice Conversion")
    print("=" * 50)
    print(f"  Device:     {args.device}")
    print(f"  Model:      {args.model} ({MODEL_PATHS[args.model]['config']})")
    print(f"  Target spk: {args.target_spk}")
    print(f"  Mode:       {args.mode}")
    print("=" * 50)

    vc = VCRunner(target_wav=args.target_spk, device=args.device, model=args.model)

    if args.mode == "realtime":
        run_realtime(vc)
    elif args.mode == "file":
        if not args.input:
            print("ERROR: --input required for file mode")
            sys.exit(1)
        vc.process_file(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
