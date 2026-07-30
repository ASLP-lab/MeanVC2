"""
Extract BN features (40ms/frame) from wav files using JIT Fast-U2++.
Usage: python extract_bn_80ms.py --input_dir /path/to/wavs --output_dir /path/to/bns

Output: BN at 40ms per frame (same as extract_bn_160ms).
"""
import os, sys, argparse
import numpy as np
import torch
import soundfile as sf
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# JIT ASR parameters
BN_WINDOW = 11   # fbank frames per JIT call
BN_STRIDE = 8    # fbank advance per call → 40ms per BN frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='wav file directory')
    parser.add_argument('--output_dir', required=True, help='BN .npy output directory')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    asr_ckpt = os.path.join(_ROOT, "preprocess/ckpts/fastu2pp_80ms.pt")
    asr = torch.jit.load(asr_ckpt)
    asr.eval()
    print(f"JIT ASR model loaded")

    wav_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.wav')])
    print(f"Found {len(wav_files)} wav files")

    for wf in tqdm(wav_files, desc="Extracting BN"):
        uid = wf.replace('.wav', '')
        out_path = os.path.join(args.output_dir, f"{uid}.npy")
        if os.path.exists(out_path):
            continue

        wav, sr = sf.read(os.path.join(args.input_dir, wf), dtype='int16')
        if wav.ndim > 1:
            wav = wav.mean(axis=1).astype(np.int16)
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        if sr != 16000:
            import librosa
            wav = librosa.core.resample(wav, orig_sr=sr, target_sr=16000, res_type='soxr_hq')

        wav_t = torch.from_numpy(wav * (1 << 15)).unsqueeze(0)
        fbanks = kaldi.fbank(wav_t, frame_length=25, frame_shift=10,
                             snip_edges=True, num_mel_bins=80,
                             energy_floor=0.0, dither=0.0,
                             sample_frequency=16000).squeeze(0)

        offset = 4
        att_cache = torch.zeros(6, 4, 4, 128)
        cnn_cache = torch.zeros(6, 1, 256, 8)

        bns = []
        i = 0
        while i + BN_WINDOW <= fbanks.shape[0]:
            fb = fbanks[i:i+BN_WINDOW].unsqueeze(0)
            out, att_cache, cnn_cache = asr(
                fb,
                torch.tensor(offset, dtype=torch.int64),
                torch.tensor(4, dtype=torch.int64),
                att_cache, cnn_cache,
            )
            bns.append(out.squeeze(0).detach())
            offset += 2
            i += BN_STRIDE

        bn = torch.cat(bns, dim=0).numpy()
        np.save(out_path, bn)

    print(f"Done: {len(wav_files)} files → {args.output_dir}")


if __name__ == "__main__":
    main()
