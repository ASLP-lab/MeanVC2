
<div align="center">

<h1>🎤 MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion</h1>


![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

[![GitHub](https://img.shields.io/badge/GitHub-MeanVC2-181717?logo=github&logoColor=white)](https://github.com/ASLP-lab/MeanVC2)
[![Demo Page](https://img.shields.io/badge/GitHub-Demo--Page-8A2BE2?logo=github&logoColor=white&labelColor=181717)](https://aslp-lab.github.io/MeanVC2/)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-FF9D00)](https://huggingface.co/ASLP-lab/MeanVC2)
[![Lab](https://img.shields.io/badge/🏫%20ASLP-Lab-4A90D9)](http://www.npu-aslp.org/)

<p>
    Guobin Ma<sup>1,*</sup>,
    Yuxuan Xia<sup>1,*</sup>,
    Yuepeng Jiang<sup>1</sup>,
    Dake Guo<sup>1</sup>,
    Hanke Xie<sup>1</sup>,
    Jingbin Hu<sup>1</sup>,
    Yanbo Wang<sup>2</sup>,
    Lei Xie<sup>1,**</sup>,
    Pengcheng Zhu<sup>3,**</sup>
</p>

<p>
    <sup>1</sup> Audio, Speech and Language Processing Group (ASLP@NPU), School of Software, Northwestern Polytechnical University, China<br>
    <sup>2</sup> The University of New South Wales, Australia<br>
    <sup>3</sup> WeNet Open Source Community, China
</p>



</div>

## 📖 Introduction

**MeanVC2** is a robust, low-latency streaming zero-shot voice conversion (VC) system built upon the diffusion-based conditional flow matching (CFM) framework. It addresses key limitations of its predecessor MeanVC, including training inefficiency, quality degradation under small-chunk settings, and sensitivity to low-quality reference audio.

By introducing **Future-Receptive Chunking (FRC)** and a **Universal Timbre Token Encoder (UTTE)**, MeanVC2 achieves high-fidelity voice conversion with an **end-to-end pipeline latency of only 110 ms** — nearly halving the 211 ms latency of MeanVC(160ms) — while maintaining superior speaker similarity and audio naturalness even with a **40 ms chunk size**. It operates under a recognition-synthesis paradigm, where a streaming ASR module extracts content representations (BNFs), and a DiT-based decoder generates target mel-spectrograms conditioned on timbre-aware features retrieved via UTTE.

## ✨ Key Features

- **Ultra-low latency streaming**: 110 ms end-to-end first-packet latency with 40 ms chunk size; full pipeline RTF < 0.633 on single CPU core.
- **Future-Receptive Chunking (FRC)**: Enables stable short-chunk conversion by explicitly scheduling past/future receptive fields across DiT layers, eliminating clean-chunk teacher forcing and reducing peak GPU memory by ~60%.
- **Universal Timbre Token Encoder (UTTE)**: Decouples fine-grained timbre extraction from direct reference mel-spectrograms, using global speaker embeddings + cross-attention to improve robustness under low-quality references and enhance zero-shot speaker similarity.
- **Mean Flows + 1-NFE inference**: Single-step ODE solving for high-quality mel-spectrogram synthesis, balancing efficiency and fidelity.
- **Lightweight yet powerful**: Only 18M parameters — comparable to MeanVC (14M) and far smaller than competing streaming VC systems.
- **End-to-end & streaming-ready**: Supports both file-based conversion and real-time microphone streaming with pre-extracted or on-the-fly features.

## 🚀 Quick Start

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/ASLP-lab/MeanVC2.git
cd MeanVC2

# Create conda environment
conda create -n meanvc2 python=3.11
conda activate meanvc2

# Install PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pretrained Models

```bash
# Download all models (preprocessing + VC + vocoder)
python initialization.py --task all
```

Or download only what you need:

```bash
python initialization.py --task preprocess   # BN + SpkEmb extraction only
python initialization.py --task train_120ms  # preprocess + 120ms VC + vocoder
python initialization.py --task train_40ms   # preprocess + 40ms VC + vocoder
```

FunASR models (Paraformer, VAD, punctuation) will auto-download from ModelScope at first use.

## 📁 Data Preparation

### Step 1: Extract Mel Spectrograms

```bash
python preprocess/extract_mel.py --input_dir /path/to/wavs --output_dir /path/to/mels
```

### Step 2: Extract Content Features (BN)

```bash
# 80ms chunk JIT (fastu2pp_80ms.pt, 11-frame window, stride=8)
python preprocess/extract_bn_80ms.py --input_dir /path/to/wavs --output_dir /path/to/bns

# 160ms chunk JIT (fastu2pp_160ms.pt, 19-frame window, stride=16)
python preprocess/extract_bn_160ms.py --input_dir /path/to/wavs --output_dir /path/to/bns
```

### Step 3: Extract Speaker Embeddings

```bash
python preprocess/extract_spk_emb.py --input_dir /path/to/wavs --output_dir /path/to/xvectors
```

### Step 4: Create Training Filelist

```bash
python scripts/create_filelist.py \
    --bn-dir /path/to/bns \
    --mel-dir /path/to/mels \
    --xvector-dir /path/to/xvectors \
    --output train.list
```

Filelist format (one line per utterance):
```
utt_id|/path/to/bn/utt_id.npy|/path/to/mel/utt_id.npy|/path/to/xvector/utt_id.npy
```

## 🎵 Inference

### Zero-Shot (Non-Streaming)

```bash
# 120ms chunk + 40ms future (recommended for quality)
python src/infer/infer_zero_shot.py \
    --model-config src/config/config_120ms_40ms.json \
    --ckpt-path ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors \
    --vocoder-ckpt-path ckpts/vocos/vocos.pt \
    --output-dir output/ \
    --file-scp test.lst \
    --bn-path /path/to/bn \
    --spk-emb-path /path/to/spk_emb \
    --chunk-size 12 --steps 3
```

### End-to-End (Single Script)

No pre-extracted features needed — input two wavs, output converted audio:

```bash
# 120ms+40ms model (recommended for quality)
python src/infer/infer_e2e.py \
    --source-wav /path/to/source.wav \
    --target-wav /path/to/target.wav \
    --ckpt-path ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors \
    --model-config src/config/config_120ms_40ms.json \
    --vocoder-ckpt-path ckpts/vocos/vocos.pt \
    --chunk-size 12 --steps 3 \
    --output-wav output.wav

# 40ms+40ms model (lower latency)
python src/infer/infer_e2e.py \
    --source-wav /path/to/source.wav \
    --target-wav /path/to/target.wav \
    --ckpt-path ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors \
    --model-config src/config/config_40ms_40ms.json \
    --vocoder-ckpt-path ckpts/vocos/vocos.pt \
    --chunk-size 4 --steps 3 \
    --output-wav output.wav
```

### Real-Time Streaming

```bash
# File mode
cd runtime
python run_rt.py --mode file --input in.wav --output out.wav --model 120ms

# Microphone mode
python run_rt.py --mode realtime --model 40ms
```

## 🏗️ Model Architecture

MeanVC2 consists of five core components:

| Component | Description |
|-----------|-------------|
| **Streaming ASR Encoder** | Fast-U2++ (WeNet) extracts bottleneck features (BNFs) from source waveform; 80 ms chunk size for streaming inference |
| **Speaker Encoder** | ECAPA-TDNN + WavLM upstream extracts a global speaker embedding from reference audio |
| **Universal Timbre Token Encoder (UTTE)** | Transforms global speaker embedding into K key-value UTT pairs; BNFs serve as queries in cross-attention to retrieve fine-grained, pronunciation-aware timbre cues |
| **DiT-based CFM Decoder** | 4-layer DiT (hidden dim 512, 2 heads) with Future-Receptive Chunking (FRC); trained with mean flows objective for 1-NFE mel-spectrogram generation |
| **Vocoder** | Vocos converts mel-spectrograms to 16 kHz high-fidelity speech waveforms |

**Total parameters**: ~18M

## 🏋️ Training

```bash
# 120ms+40ms training (recommended for quality)
bash scripts/train_120ms_40ms.sh 0                     # single GPU
bash scripts/train_120ms_40ms.sh "0,1,2,3,4,5,6,7"    # 8 GPUs

# 40ms+40ms training (lower latency)
bash scripts/train_40ms_40ms.sh 0                      # single GPU
bash scripts/train_40ms_40ms.sh "0,1,2,3,4,5,6,7"     # 8 GPUs

# Override dataset and experiment name
DATASET_PATH=/path/to/train.list EXP_NAME=my_exp bash scripts/train_120ms_40ms.sh "0,1,2,3"
```

The dataset path expects a `.list` file (line items: `utt_id|/path/to/bn.npy|/path/to/mel.npy|/path/to/xvector.npy`), generated by `scripts/create_filelist.py`.

## 🙏 Acknowledgements

This work builds upon the following open-source projects:

- [F5-TTS](https://github.com/SWivid/F5-TTS) — DiT-based CFM backbone
- [Vocos](https://github.com/gemelo-ai/vocos) — Neural vocoder
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) — Speaker encoder upstream
- [ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN) — Speaker embedding model
- [WeNet](https://github.com/wenet-e2e/wenet) — ASR encoder for BN extraction
- [s3prl](https://github.com/s3prl/s3prl) — Self-supervised speech models
- [FunASR](https://github.com/modelscope/FunASR) — ASR for CER evaluation
- [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) — Training data

## 📜 License & Disclaimer

MeanVC2 is released under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0). This open-source license allows you to freely use, modify, and distribute the model, as long as you include the appropriate copyright notice and disclaimer.

MeanVC2 is designed for research and legitimate applications in voice conversion technology. Users must obtain proper consent from individuals whose voices are being converted or used as references. We strongly discourage any malicious use including impersonation, fraud, or creating misleading audio content. Users are solely responsible for ensuring their use cases comply with ethical standards and legal requirements.

## 📄 Citation

If you find our work helpful, please cite:

```bibtex
@article{ma2026meanvc2,
  title={MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion},
  author={Ma, Guobin and Xia, Yuxuan and Jiang, Yuepeng and Guo, Dake and Xie, Hanke and Hu, Jingbin and Wang, Yanbo and Xie, Lei and Zhu, Pengcheng},
  journal={arXiv preprint arXiv:2606.09050},
  year={2026}
}
```

## ✉️ Contact

For questions or collaborations, please contact: guobin.ma@mail.nwpu.edu.cn or lxie@nwpu.edu.cn

## ⭐ Star History


