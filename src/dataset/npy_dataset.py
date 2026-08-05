"""
Simple file-based dataset for MeanVC2 training.

Format of file list (one line per utterance):
  utt|bn_path|mel_path|xvector_path

All fields are required and point to .npy files.

BN is stored at 40ms/frame; mel at 10ms/frame.
BN is interpolated 4x in __getitem__ so both have the same frame rate
before entering the batch — matching inference behaviour.
"""
import numpy as np
import torch
import torch.nn.functional as F
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

        # Load BN (40ms rate) → interpolate 4x to match mel rate (10ms)
        bn = torch.from_numpy(np.load(sample["bn"])).float()          # (T_bn, 256)
        bn = bn.unsqueeze(0).transpose(1, 2)                          # (1, 256, T_bn)
        bn = F.interpolate(bn, size=int(bn.shape[2] * 4),
                           mode='linear', align_corners=True)         # (1, 256, T_bn*4)
        bn = bn.transpose(1, 2).squeeze(0)                            # (T_bn*4, 256)
        features["bn"] = bn

        # Load mel (10ms rate, same frame rate as interpolated BN)
        mel = torch.from_numpy(np.load(sample["mel"])).float()        # (T_mel, 80)
        features["mel"] = mel

        # Load xvector
        xvector = torch.from_numpy(np.load(sample["xvector"])).float().squeeze()
        features["xvector"] = xvector                                    # (256,)

        # Truncate to max_len (shorter of bn / mel)
        min_len = min(bn.shape[0], mel.shape[0], self.max_len)
        features["bn"] = features["bn"][:min_len]
        features["mel"] = features["mel"][:min_len]
        features["inputs_length"] = torch.tensor(min_len).long()

        return features

    def custom_collate_fn(self, batch):
        """Pad features to max length in batch using configured pad values."""
        feature_list = self.feature_list + self.additional_feature_list + ["inputs_length"]
        max_len = max(b["mel"].size(0) for b in batch)

        out = {}
        for key in feature_list:
            if key == "inputs_length":
                out[key] = torch.stack([b[key] for b in batch])
            elif key == "xvector":
                out[key] = torch.stack([b[key] for b in batch])            # (B, 256)
            else:
                val_list = []
                for b in batch:
                    val = b[key]
                    pad_len = max_len - val.size(0)
                    if pad_len > 0:
                        val = F.pad(val, (0, 0, 0, pad_len), value=0.0)
                    val_list.append(val)
                out[key] = torch.stack(val_list)
        return out
