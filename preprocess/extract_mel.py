"""
Extract mel spectrograms from wav files.
Usage: python extract_mel.py --input_dir /path/to/wavs --output_dir /path/to/mels [--config preConfiged16K_10ms]

Uses the same mel extraction pipeline as the original Lance-based preprocessing:
  STFT → mel filterbank → amp_to_db → ref_level_db → normalize → [-1, 1]
"""
import os, sys, argparse
import numpy as np
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import preprocess.config as cfg_module
from preprocess.audio import load_wav, melspectrogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing wav files')
    parser.add_argument('--output_dir', required=True, help='Output directory for mel .npy files')
    parser.add_argument('--config', default='preConfiged16K_10ms',
                        help='Preset name: preConfiged16K_10ms, preConfiged24K, etc.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = getattr(cfg_module, args.config)  # Already an instance

    wav_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.wav')])
    print(f"Found {len(wav_files)} wav files, config: {args.config}")

    for wf in tqdm(wav_files, desc="Extracting Mel"):
        uid = wf.replace('.wav', '')
        out_path = os.path.join(args.output_dir, f"{uid}.npy")
        if os.path.exists(out_path):
            continue

        wav_path = os.path.join(args.input_dir, wf)
        wav = load_wav(wav_path, cfg.sample_rate, cfg.win_size, cfg.hop_size)

        # Identical to legacy __extract_mel(): melspectrogram + aligned truncation
        n_samples = len(wav)
        mel = melspectrogram(wav, cfg).astype(np.float32)  # (80, T), range [-1, 1]
        n_frames = mel.shape[1]
        desired_frames = int(min(n_samples / cfg.hop_size, n_frames))
        mel = mel[:, :desired_frames]  # Align to hop_size boundary
        mel = mel.T  # (T, 80) so NpyDataset can feed directly into the model after loading

        np.save(out_path, mel)

    print(f"Done: {len(wav_files)} files → {args.output_dir}")


if __name__ == "__main__":
    main()
