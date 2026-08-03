"""
End-to-end inference: source audio + target speaker audio -> converted wav.
No need to pre-extract BN/spk_emb; everything is done within this script.

Usage:
  # 120ms+40ms model (ASR=160ms)
  python src/infer/infer_e2e.py --model 120ms \
    --source-wav /path/to/source.wav --target-wav /path/to/target.wav \
    --output-wav output.wav --steps 3

  # 40ms+40ms model (ASR=80ms)
  python src/infer/infer_e2e.py --model 40ms \
    --source-wav /path/to/source.wav --target-wav /path/to/target.wav \
    --output-wav output.wav --steps 3
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import torchaudio
import soundfile as sf
import torchaudio.compliance.kaldi as kaldi

# Add meanvc2 root directory
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src/infer"))

from dit_kvcache import DiT
from src.model.utils import load_checkpoint


# ---------------------------------------------------------------------------
# Per-model presets: ASR + VC paths and parameters
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "120ms": {
        # ASR
        "asr_ckpt": os.path.join(_ROOT, "preprocess/ckpts/fastu2pp_160ms.pt"),
        "bn_window": 19,
        "bn_stride": 16,
        "offset_init": 8,
        "offset_step": 4,
        "req_cache": 8,
        # VC
        "vc_ckpt": os.path.join(_ROOT, "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
        "vc_config": os.path.join(_ROOT, "src/config/config_120ms_40ms.json"),
        "chunk_size": 12,
        "block_size": 4,
    },
    "40ms": {
        # ASR
        "asr_ckpt": os.path.join(_ROOT, "preprocess/ckpts/fastu2pp_80ms.pt"),
        "bn_window": 11,
        "bn_stride": 8,
        "offset_init": 4,
        "offset_step": 2,
        "req_cache": 4,
        # VC
        "vc_ckpt": os.path.join(_ROOT, "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors"),
        "vc_config": os.path.join(_ROOT, "src/config/config_40ms_40ms.json"),
        "chunk_size": 4,
        "block_size": 4,
    },
}


# ---------------------------------------------------------------------------
# ASR BN extraction
# ---------------------------------------------------------------------------

def load_asr_model(asr_ckpt_path, device):
    model = torch.jit.load(asr_ckpt_path, map_location=device)
    model.eval()
    return model


def extract_bn(wav_path, asr_model, *, window, stride,
               offset_init, offset_step, req_cache):
    """Extract BN features from wav using JIT ASR with sliding window."""
    wav, sr = sf.read(wav_path, dtype='int16')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    if sr != 16000:
        import librosa
        wav = librosa.core.resample(wav, orig_sr=sr, target_sr=16000, res_type='soxr_hq')

    wav_tensor = torch.from_numpy(wav * (1 << 15)).unsqueeze(0)
    fbanks = kaldi.fbank(wav_tensor, frame_length=25, frame_shift=10,
                         snip_edges=True, num_mel_bins=80, energy_floor=0.0, dither=0.0,
                         sample_frequency=16000).squeeze(0)

    offset = offset_init
    att_cache = torch.zeros(6, 4, req_cache, 128)
    cnn_cache = torch.zeros(6, 1, 256, 8)

    bns, i, T = [], 0, fbanks.shape[0]
    while i + window <= T:
        fb = fbanks[i:i + window].unsqueeze(0)
        out, att_cache, cnn_cache = asr_model(
            fb,
            torch.tensor(offset, dtype=torch.int64),
            torch.tensor(req_cache, dtype=torch.int64),
            att_cache, cnn_cache,
        )
        bns.append(out.squeeze(0).detach())
        offset += offset_step
        i += stride

    return torch.cat(bns, dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# Speaker embedding extraction
# ---------------------------------------------------------------------------

def load_spk_model(device):
    from preprocess.models.ecapa_tdnn import ECAPA_TDNN_SMALL
    ckpt_path = os.path.join(_ROOT, "preprocess/ckpts/wavlm_large_finetune.pth")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=False)
    model.to(device)
    model.eval()
    return model


def extract_spk_emb(wav_path, spk_model, device):
    wav, sr = sf.read(wav_path, dtype='int16')
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.int16)
    wav = wav / np.iinfo(wav.dtype).max
    if sr != 16000:
        import librosa
        wav = librosa.core.resample(wav, orig_sr=sr, target_sr=16000, res_type='soxr_hq')
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = spk_model(wav_tensor)
    return emb.squeeze().detach().cpu().numpy()


# ---------------------------------------------------------------------------
# VC inference (streaming-style on pre-extracted BN)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def vc_inference(model, vocos, bn, spk_emb, chunk_size, block_size, steps, device):
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    bn_t = torch.from_numpy(bn).to(device).unsqueeze(0).transpose(1, 2)
    bn_t = torch.nn.functional.interpolate(
        bn_t, size=int(bn_t.shape[2] * 4), mode='linear', align_corners=True
    ).transpose(1, 2)
    spk_t = torch.from_numpy(spk_emb).to(device)
    if len(spk_t.shape) == 1:
        spk_t = spk_t.unsqueeze(0)

    seq_len = bn_t.shape[1]
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    x_pred_collect, offset, kv_cache = [], 0, None
    noise = torch.randn(1, seq_len, 80, device=device, dtype=bn_t.dtype)

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        mid = min(start + chunk_size, seq_len)
        end = min(mid + block_size, seq_len)
        curr_len = mid - start
        cond_in = torch.cat([bn_t[:, start:mid], bn_t[:, mid:end]], dim=1)
        x = noise[:, start:end]

        for i in range(steps):
            t, r = timesteps[i], timesteps[i + 1]
            t_t = torch.full((x.size(0),), t, device=x.device)
            r_t = torch.full((x.size(0),), r, device=x.device)
            u, tmp_kv = model(x, t_t, r_t, cache=None, cond=cond_in, spks=spk_t,
                              offset=offset, is_inference=True, kv_cache=kv_cache)
            x = x - (t - r) * u
        kv_cache = tmp_kv
        x_pred_collect.append(x[:, :curr_len])
        offset += curr_len

    x_pred = torch.cat(x_pred_collect, dim=1)
    mel = x_pred.transpose(1, 2)
    mel = (mel + 1) / 2
    wav = vocos.decode(mel)
    return mel, wav


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeanVC2 End-to-End Inference")
    parser.add_argument("--model", type=str, default="120ms", choices=["120ms", "40ms"],
                        help="Model preset (auto-sets ASR+VC paths and params)")
    parser.add_argument("--source-wav", type=str, required=True)
    parser.add_argument("--target-wav", type=str, required=True)
    parser.add_argument("--output-wav", type=str, default="output.wav")
    parser.add_argument("--output-mel", type=str, default=None)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")

    # Overrides (optional — defaults come from --model preset)
    parser.add_argument("--asr-ckpt", type=str, default=None)
    parser.add_argument("--bn-window", type=int, default=None)
    parser.add_argument("--bn-stride", type=int, default=None)
    parser.add_argument("--vc-ckpt", type=str, default=None)
    parser.add_argument("--vc-config", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--vocoder-ckpt-path", type=str,
                        default=os.path.join(_ROOT, "ckpts/vocos/vocos.pt"))
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    asr_ckpt = args.asr_ckpt or preset["asr_ckpt"]
    bn_window = args.bn_window or preset["bn_window"]
    bn_stride = args.bn_stride or preset["bn_stride"]
    offset_init = preset["offset_init"]
    offset_step = preset["offset_step"]
    req_cache = preset["req_cache"]
    vc_ckpt = args.vc_ckpt or preset["vc_ckpt"]
    vc_config = args.vc_config or preset["vc_config"]
    chunk_size = args.chunk_size or preset["chunk_size"]
    block_size = preset["block_size"]
    device = args.device

    print(f"[Model] {args.model}  ASR={os.path.basename(asr_ckpt)}"
          f"  window={bn_window}  stride={bn_stride}  chunk={chunk_size}")

    print(f"[1/5] Loading ASR model (JIT)...")
    asr_model = load_asr_model(asr_ckpt, "cpu")

    print(f"[2/5] Loading speaker model...")
    spk_model = load_spk_model(device)

    print(f"[3/5] Loading VC + Vocos models...")
    with open(vc_config) as f:
        model_config = json.load(f)
    vc_model = DiT(**model_config["model"]).to(device)
    vc_model = load_checkpoint(vc_model, vc_ckpt, device=device, use_ema=True)
    vc_model = vc_model.float().eval()
    vocos = torch.jit.load(args.vocoder_ckpt_path).to(device)

    print(f"[4/5] Extracting features...")
    t0 = time.time()
    bn = extract_bn(args.source_wav, asr_model,
                    window=bn_window, stride=bn_stride,
                    offset_init=offset_init, offset_step=offset_step,
                    req_cache=req_cache)
    spk_emb = extract_spk_emb(args.target_wav, spk_model, device)
    print(f"   BN: {bn.shape}  SpkEmb: {spk_emb.shape}  time: {time.time() - t0:.1f}s")

    print(f"[5/5] VC inference ({args.steps}-step)...")
    t0 = time.time()
    mel, wav = vc_inference(vc_model, vocos, bn, spk_emb,
                            chunk_size, block_size, args.steps, device)
    print(f"   time: {time.time() - t0:.1f}s  output: {wav.shape[-1] / 16000:.1f}s")

    sf.write(args.output_wav, wav.cpu().squeeze().numpy(), 16000)
    print(f"   Saved: {args.output_wav}")

    if args.output_mel:
        np.save(args.output_mel, mel.cpu().numpy())
        print(f"   Saved: {args.output_mel}")
