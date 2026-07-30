"""
MeanVC2 Checkpoint Download Script

Downloads required checkpoints from HuggingFace.

Usage:
    python initialization.py --task preprocess      # BN + SpkEmb extraction
    python initialization.py --task train_120ms      # preprocess + 120ms_40ms VC model + vocoder
    python initialization.py --task train_40ms       # preprocess + 40ms_40ms VC model + vocoder
    python initialization.py --task all              # everything
"""

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "ASLP-lab/MeanVC2"

# ---- file mapping: (hf_filename, local_path) ----
# HF repo has flat structure; local code expects nested directories.

# ASR encoder (Fast-U2++)
ASR_FILES = [
    ("fastu2pp_80ms.pt",  "preprocess/ckpts/fastu2pp_80ms.pt"),
    ("fastu2pp_160ms.pt", "preprocess/ckpts/fastu2pp_160ms.pt"),
]

# Speaker embedding: WavLM + ECAPA-TDNN
# These are existing public models — NOT in our HF repo.
# wavlm_large.pt          → Microsoft WavLM-Large (public)
# wavlm_large_cfg.pt      → extracted from wavlm_large.pt (tiny config, ~10 KB)
# wavlm_large_finetune.pth → Google Drive (ECAPA-TDNN fine-tuned weights)
SPK_FILES = [
    # download handled separately by _download_wavlm()
]

# VC model: 120ms chunk + 40ms future (recommended for quality)
VC_MODEL_120MS = [
    ("meanvc2_120ms_40ms.safetensors", "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
]

# VC model: 40ms chunk + 40ms future (lower latency)
VC_MODEL_40MS = [
    ("meanvc2_40ms_40ms.safetensors", "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors"),
]

# Vocoder (Vocos)
VOCODER_FILES = [
    ("vocos.pt",      "ckpts/vocos/vocos.pt"),
]

TASK_FILES = {
    "preprocess":  ASR_FILES + SPK_FILES,
    "train_120ms": ASR_FILES + SPK_FILES + VC_MODEL_120MS + VOCODER_FILES,
    "train_40ms":  ASR_FILES + SPK_FILES + VC_MODEL_40MS + VOCODER_FILES,
    "all":         ASR_FILES + SPK_FILES + VC_MODEL_120MS + VC_MODEL_40MS + VOCODER_FILES,
}


# ---------------------------------------------------------------------------
# WavLM download (existing public models, NOT in our HF repo)
# ---------------------------------------------------------------------------

WAVLM_LARGE_URL = (
    "https://github.com/microsoft/unilm/releases/download/wavlm/WavLM-Large.pt"
)

WAVLM_FINETUNED_URL = (
    "https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view"
)


def _download_wavlm():
    """
    Download wavlm_large.pt from Microsoft's public release and extract
    wavlm_large_cfg.pt from it.  For wavlm_large_finetune.pth, print the
    Google Drive link (can't be auto-downloaded).
    """
    import urllib.request

    wavlm_path = "preprocess/ckpts/wavlm_large.pt"
    cfg_path = "preprocess/ckpts/wavlm_large_cfg.pt"
    finetuned_path = "preprocess/ckpts/wavlm_large_finetune.pth"

    # --- wavlm_large.pt ---
    if not os.path.exists(wavlm_path):
        print("  [download] wavlm_large.pt from Microsoft WavLM-Large public release ...")
        try:
            os.makedirs("preprocess/ckpts", exist_ok=True)
            urllib.request.urlretrieve(WAVLM_LARGE_URL, wavlm_path)
            print(f"  [done]  {wavlm_path}")
        except Exception as e:
            print(f"  [warn]  auto-download failed: {e}")
            print(f"  [manual] Download WavLM-Large.pt from:")
            print(f"          {WAVLM_LARGE_URL}")
            print(f"          and save as {wavlm_path}")
            return  # can't extract cfg without the base model
    else:
        print(f"  [skip]  {wavlm_path}")

    # --- wavlm_large_cfg.pt (extract from wavlm_large.pt) ---
    if not os.path.exists(cfg_path):
        print("  [extract] wavlm_large_cfg.pt from wavlm_large.pt ...")
        try:
            checkpoint = __import__("torch").load(wavlm_path, map_location="cpu")
            cfg_dict = checkpoint.get("cfg", checkpoint.get("config"))
            if cfg_dict is None:
                print("  [warn]  could not find 'cfg' key in wavlm_large.pt")
            else:
                os.makedirs("preprocess/ckpts", exist_ok=True)
                __import__("torch").save(cfg_dict, cfg_path)
                print(f"  [done]  {cfg_path}")
        except Exception as e:
            print(f"  [warn]  config extraction failed: {e}")
            print(f"  [manual] Extract the 'cfg' key from wavlm_large.pt and save as {cfg_path}")
    else:
        print(f"  [skip]  {cfg_path}")

    # --- wavlm_large_finetune.pth ---
    if not os.path.exists(finetuned_path):
        print(f"  [manual] wavlm_large_finetune.pth must be downloaded from Google Drive:")
        print(f"          {WAVLM_FINETUNED_URL}")
        print(f"          and saved as {finetuned_path}")
    else:
        print(f"  [skip]  {finetuned_path}")


# ---------------------------------------------------------------------------

def download_files(task: str):
    files = TASK_FILES[task]
    print(f"Task: {task} | {len(files)} file(s) from HF")

    for hf_filename, local_path in files:
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(local_path):
            print(f"  [skip] {local_path}")
            continue

        print(f"  [download] {hf_filename} -> {local_path} ...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_filename,
        )
        shutil.copy(downloaded, local_path)
        print(f"  [done]  {local_path}")

    print("\n--- HF downloads complete ---\n")

    # WavLM (existing public models, from external sources)
    _download_wavlm()

    # FunASR note
    if task != "preprocess":
        print("\nNote: FunASR models (Paraformer, VAD, punctuation) auto-download from ModelScope at first use.")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MeanVC2 checkpoints from HuggingFace"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_FILES.keys()),
        help="'preprocess' for BN+SpkEmb extraction; "
             "'train_120ms' adds 120ms VC model + vocoder; "
             "'train_40ms' adds 40ms VC model + vocoder; "
             "'all' for everything",
    )
    args = parser.parse_args()
    download_files(args.task)
