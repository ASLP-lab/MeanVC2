# Copyright 2019 The NWPU-ASLP TTS/VC team. All Rights Reserved.
#
# Author       : Yang Shan 
# Email        : syang.mix@gmail.com
# Filename     : audio.py
# Created time : 2019-10-21 19:42
# Last modified: 2020-01-03 12:06
# ==============================================================================
"""Definitions of all acoustic extraction/reconstruction functions."""
import torch
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf


def load_wav(wav_path, target_sr, win_size, hop_size, allow_upsample=True):
    audio, raw_sr = librosa.core.load(wav_path, sr=None)
    if raw_sr < target_sr and not allow_upsample:
        raise Exception(f"Resample from {raw_sr} to {target_sr} is not allowed. Please check sample rate or set `allow_upsample=True`. wav: {wav_path}")
    audio = librosa.core.resample(
        audio, 
        orig_sr=raw_sr,
        target_sr=target_sr, 
        res_type='kaiser_best')
    target_length = (audio.size // hop_size) * hop_size # + win_size // hop_size) * hop_size
    pad_len = (target_length - audio.size) // 2
    if pad_len < 0:
        audio = audio[:target_length]
    else:
        if audio.size % 2 == 0:
            audio = np.pad(audio, (pad_len, pad_len), mode='reflect')
        else:
            audio = np.pad(audio, (pad_len, pad_len + 1), mode='reflect')
    return audio

mel_basis = {}
hann_window = {}

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec
def save_wav(wav, path, hparams, norm=False):
    if norm:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))
    else:
        sf.write(path, wav, hparams.sample_rate)


def preemphasis(x, preemphasis=0.97):
    return signal.lfilter([1, -preemphasis], [1], x)


def inv_preemphasis(x, preemphasis=0.97):
    return signal.lfilter([1], [1, -preemphasis], x)


def trim_silence(wav, hparams):
    '''
    Trim leading and trailing silence
    '''
    # These params are separate and tunable per dataset.
    unused_trimed, index = librosa.effects.trim(wav,
                                                top_db=hparams.trim_top_db,
                                                frame_length=hparams.
                                                trim_fft_size,
                                                hop_length=hparams.
                                                trim_hop_size)
    num_sil_samples = int(
        hparams.num_silent_frames * hparams.hop_size)
    # head silence is set as half of num_sil_samples
    start_idx = max(index[0] - int(num_sil_samples//2), 0)
    # tail silence is set as twice of num_sil_samples
    stop_idx = min(index[1] + num_sil_samples*2, len(wav))
    trimmed = wav[start_idx:stop_idx]
    if hparams.pad_sli:
        sli = [0.0 for i in range(hparams.pad_sli_frame * hparams.hop_size)]
        trimmed = np.concatenate((sli,trimmed,sli),axis=0)
    return trimmed


def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    frame_shift_ms = 1000 * (hop_size / hparams.sample_rate)

    # To ensure the frame_shift as 10ms or 12.5ms
    assert (frame_shift_ms == 12.5 or frame_shift_ms == 10)
    window_size = 1000 * (hparams.win_size / hparams.sample_rate)

    # To ensure the windows size as 50ms
    #assert window_size == 50

    return hop_size


def linearspectrogram(wav, hparams):
    # D = _stft(preemphasis(wav, hparams.preemphasis), hparams)
    D = _stft(wav, hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def melspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams),
                   hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def linear_to_mel(linear_spectrogram, hparams):
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram
    linear = _db_to_amp(D + hparams.ref_level_db)
    S = _amp_to_db(_linear_to_mel(np.abs(linear), hparams),
                   hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def inv_linear_spectrogram(linear_spectrogram, hparams):
    '''Converts linear spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    # Convert back to linear
    S = _db_to_amp(D + hparams.ref_level_db)

    return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis)


def inv_mel_spectrogram(mel_spectrogram, hparams):
    '''Converts mel spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    # Convert back to linear
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)

    return _griffin_lim(S ** hparams.power, hparams)


def _griffin_lim(S, hparams):
    '''
    librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=get_hop_size(hparams),
                        win_length=hparams.win_size)


def _istft(y, hparams):
    return librosa.istft(y,
                         hop_length=get_hop_size(hparams),
                         win_length=hparams.win_size)


def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    '''compute right padding (final frame)
    '''
    return int(fsize // 2)


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(sr=hparams.sample_rate,
                               n_fft=hparams.n_fft,
                               n_mels=80,
                               fmin=hparams.fmin,
                               fmax=hparams.fmax)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_acoustic:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_db) /
                                                          (-hparams.min_db)) -
                           hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_db) /
                                                    (-hparams.min_db)),
                           0, hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_db >= 0
    if hparams.symmetric_acoustic:
        return ((2 * hparams.max_abs_value) *
                ((S - hparams.min_db) / (-hparams.min_db)) -
                hparams.max_abs_value)
    else:
        return (hparams.max_abs_value *
                ((S - hparams.min_db) / (-hparams.min_db)))


def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_acoustic:
            return (((np.clip(D, -hparams.max_abs_value,
                              hparams.max_abs_value) + hparams.max_abs_value)
                     * -hparams.min_db / (2 * hparams.max_abs_value))
                    + hparams.min_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_db /
                     hparams.max_abs_value) + hparams.min_db)

    if hparams.symmetric_acoustic:
        return (((D + hparams.max_abs_value) * -hparams.min_db /
                 (2 * hparams.max_abs_value)) + hparams.min_db)
    else:
        return ((D * -hparams.min_db / hparams.max_abs_value) + hparams.min_db)


def _extract_barks_min_max(file_names, bark_dim):
    assert len(file_names) > 0
    bark_mins_per_file = np.zeros((len(file_names), bark_dim))
    bark_maxs_per_file = np.zeros((len(file_names), bark_dim))

    for i, file_name in enumerate(file_names):
        bark_spectrogram = np.fromfile(file_name, dtype=np.float32).reshape(-1, bark_dim)
        bark_mins_per_file[i, ] = np.amin(bark_spectrogram, axis=0)
        bark_maxs_per_file[i, ] = np.amax(bark_spectrogram, axis=0)

    bark_mins = np.asarray(np.reshape(np.amin(bark_mins_per_file, axis=0), (1, bark_dim)),
                           dtype=np.float32)
    bark_maxs = np.asarray(np.reshape(np.amax(bark_maxs_per_file, axis=0), (1, bark_dim)),
                           dtype=np.float32)

    min_max = {
        "bark_min": bark_mins,
        "bark_max": bark_maxs
    }
    return min_max


def _normalize_min_max(spec, maxs, mins, max_value=1.0, min_value=0.0):
    spec_dim = len(spec.T)
    num_frame = len(spec)

    max_min = maxs - mins
    max_min = np.reshape(max_min, (1, spec_dim))
    max_min[max_min <= 0.0] = 1.0

    target_max_min = np.zeros((1, spec_dim))
    target_max_min.fill(max_value - min_value)
    target_max_min[max_min <= 0.0] = 1.0

    spec_min = np.tile(mins, (num_frame, 1))
    target_min = np.tile(min_value, (num_frame, spec_dim))
    spec_range = np.tile(max_min, (num_frame, 1))
    norm_spec = np.tile(target_max_min, (num_frame, 1)) / spec_range
    norm_spec = norm_spec * (spec - spec_min) + target_min
    return norm_spec


def _denormalize_min_max(spec, maxs, mins, max_value=1.0, min_value=0.0):
    spec_dim = len(spec.T)
    num_frame = len(spec)

    max_min = maxs - mins
    max_min = np.reshape(max_min, (1, spec_dim))
    max_min[max_min <= 0.0] = 1.0

    target_max_min = np.zeros((1, spec_dim))
    target_max_min.fill(max_value - min_value)
    target_max_min[max_min <= 0.0] = 1.0

    spec_min = np.tile(mins, (num_frame, 1))
    target_min = np.tile(min_value, (num_frame, spec_dim))
    spec_range = np.tile(max_min, (num_frame, 1))
    denorm_spec = spec_range / np.tile(target_max_min, (num_frame, 1))
    denorm_spec = denorm_spec * (spec - target_min) + spec_min
    return denorm_spec
