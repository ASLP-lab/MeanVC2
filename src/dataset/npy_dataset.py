"""
Simple file-based dataset for MeanVC2 training.

Format of file list (one line per utterance):
  utt|bn_path|mel_path|xvector_path

All fields are required and point to .npy files.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class NpyDataset(Dataset):
    def __init__(
        self,
        filelist_path: str,
        feature_list: list[str],
        additional_feature_list: list[str] = None,
        feature_pad_values: list[float] = None,
        max_len: int = 500,
        precision: str = "fp32",
    ):
        self.feature_list = feature_list
        self.additional_feature_list = additional_feature_list or []
        self.feature_pad_values = feature_pad_values or [0.0] * len(feature_list)
        self.max_len = max_len
        self.precision = precision

        # Parse file list
        self.samples = []
        with open(filelist_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 4:
                    raise ValueError(f"Expected >=4 fields (utt|bn|mel|xvector), got {len(parts)}: {line}")
                self.samples.append({
                    "utt": parts[0],
                    "bn": parts[1],
                    "mel": parts[2],
                    "xvector": parts[3],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = {}

        # Load BN
        bn = np.load(sample["bn"])  # (T, 256) float32
        features["bn"] = torch.from_numpy(bn).float()

        # Load mel
        mel = np.load(sample["mel"])  # (T, 80) float32
        features["mel"] = torch.from_numpy(mel).float()

        # Load xvector
        xvector = np.load(sample["xvector"])  # (256,) float32
        xvector = torch.from_numpy(xvector).float().squeeze()
        features["xvector"] = xvector  # (256,)

        # input length in time frames
        features["inputs_length"] = torch.tensor(features["mel"].size(0)).long()

        return features

    @staticmethod
    def custom_collate_fn(batch):
        """Pad features to max length in batch."""
        feature_list = ["bn", "mel", "xvector", "inputs_length"]
        max_len = max(b["mel"].size(0) for b in batch)

        out = {}
        for key in feature_list:
            if key == "inputs_length":
                out[key] = torch.stack([b[key] for b in batch])
            elif key == "xvector":
                out[key] = torch.stack([b[key] for b in batch])  # (B, 256)
            else:  # bn, mel → pad along time axis
                val_list = []
                for b in batch:
                    val = b[key]
                    pad_len = max_len - val.size(0)
                    if pad_len > 0:
                        val = torch.nn.functional.pad(val, (0, 0, 0, pad_len))
                    val_list.append(val)
                out[key] = torch.stack(val_list)
        return out
