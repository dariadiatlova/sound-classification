import random
import torch
import torchaudio
from torchaudio import transforms
from typing import Tuple, Optional


def load_wav_file(audio_file_path: str) -> Optional[Tuple[torch.tensor, int]]:
    try:
        signal, sr = torchaudio.load(audio_file_path)
        return signal, sr
    except Exception:
        return


def resample_channel(audio_data: Tuple[torch.tensor, int], new_channel: int) -> Tuple[torch.tensor, int]:
    signal, sr = audio_data

    if signal.shape[0] == new_channel:
        # Nothing to do
        return audio_data

    elif new_channel == 1:
        # Convert from stereo to mono by selecting only the first channel
        re_sig = signal[:1, :]

    else:
        # Convert from mono to stereo by duplicating the first channel
        re_sig = torch.cat([signal, signal])

    return re_sig, sr


def resample(audio_data: Tuple[torch.tensor, int], new_sr: int) -> Tuple[torch.tensor, int]:
    signal, sr = audio_data

    if sr == new_sr:
        # Nothing to do
        return audio_data

    num_channels = signal.shape[0]
    # Resample first channel
    resampled_sig = torchaudio.transforms.Resample(sr, new_sr)(signal[:1, :])

    if num_channels > 1:
        # Resample the second channel and merge both channels
        resampled_two = torchaudio.transforms.Resample(sr, new_sr)(signal[1:, :])
        resampled_sig = torch.cat([resampled_sig, resampled_two])

    return resampled_sig, new_sr


def pad_trunc(audio_data: Tuple[torch.tensor, int], max_ms: int) -> Tuple[torch.tensor, int]:
    signal, sr = audio_data
    num_rows, sig_len = signal.shape
    max_len = sr // 1000 * max_ms

    if sig_len > max_len:
        # Truncate the signal to the given length
        signal = signal[:, :max_len]

    elif sig_len < max_len:
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))
        signal = torch.cat((pad_begin, signal, pad_end), 1)

    return signal, sr


def time_shift(audio_data: Tuple[torch.tensor, int], shift_limit: float) -> Tuple[torch.tensor, int]:
    signal, sr = audio_data
    _, sig_len = signal.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return signal.roll(shift_amt), sr


def spectrogram(audio_data: Tuple[torch.tensor, int],
                n_mels: int = 64, n_fft: int = 1024, hop_len=None) -> torch.tensor:
    signal, sr = audio_data
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    _spectrogram = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

    # Convert to decibels
    _spectrogram = transforms.AmplitudeToDB(top_db=top_db)(_spectrogram)
    return _spectrogram


def spectrogram_augment(spec, max_mask_pct: float = 0.1, n_freq_masks: int = 1, n_time_masks: int = 1) -> torch.tensor:
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    augment_spectrogram = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        augment_spectrogram = transforms.FrequencyMasking(freq_mask_param)(augment_spectrogram, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        augment_spectrogram = transforms.TimeMasking(time_mask_param)(augment_spectrogram, mask_value)

    return augment_spectrogram
