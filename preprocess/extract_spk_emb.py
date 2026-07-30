"""
Extract WavLM-based speaker embeddings from wav files.
Usage: python extract_spk_emb.py --input_dir /path/to/wavs --output_dir /path/to/xvectors

Matches the original Lance-based pipeline:
  sf.read (float64, no dtype) → torchaudio Resample (if needed) → ECAPA_TDNN_SMALL
"""
import os, sys, argparse
import numpy as np
import torch
import soundfile as sf
from torchaudio.transforms import Resample
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from preprocess.models.ecapa_tdnn import ECAPA_TDNN_SMALL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing wav files')
    parser.add_argument('--output_dir', required=True, help='Output directory for spk emb .npy files')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # Load model
    ckpt_path = os.path.join(_ROOT, "preprocess/ckpts/wavlm_large_finetune.pth")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=False)
    model.to(device)
    model.eval()
    print(f"SpkEmb model loaded to {device}")

    wav_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.wav')])
    print(f"Found {len(wav_files)} wav files")

    for wf in tqdm(wav_files, desc="Extracting SpkEmb"):
        uid = wf.replace('.wav', '')
        out_path = os.path.join(args.output_dir, f"{uid}.npy")
        if os.path.exists(out_path):
            continue

        # Match legacy pipeline: sf.read without dtype, returns float64 already normalized to [-1, 1]
        wav, sr = sf.read(os.path.join(args.input_dir, wf))
        wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)

        if sr != 16000:
            resample = Resample(orig_freq=sr, new_freq=16000).to(device)
            wav = resample(wav)

        with torch.no_grad():
            emb = model(wav)
        emb = emb.squeeze().detach().cpu().numpy()
        np.save(out_path, emb)

    print(f"Done: {len(wav_files)} files → {args.output_dir}")


if __name__ == "__main__":
    main()
