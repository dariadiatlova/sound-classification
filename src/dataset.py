import torch

from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from src.preprocessing.audio_metadata import get_audio_data
from src.preprocessing.audio_util_load import (resample, load_wav_file, resample_channel, pad_trunc, time_shift,
                                               spectrogram, spectrogram_augment)


class SoundDS(Dataset):
    def __init__(self, test_fold_number, test: bool = False):
        if test:
            self.audio_files, self.classes = get_audio_data(test_fold_number, test=test)
        else:
            self.audio_files, self.classes = get_audio_data(test_fold_number, test=test)
        assert len(self.audio_files) == len(self.classes), "Number of classes does not match a number of filenames."
        self.duration: int = 4000
        self.sr: int = 44100
        self.channel: int = 1
        self.shift_pct: float = 0.4

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        class_id = self.classes[idx]

        audio_data = load_wav_file(audio_file)

        if audio_data is None:
            print(audio_file)

        resampled_audio = resample(audio_data, self.sr)
        resamples_channels = resample_channel(resampled_audio, self.channel)

        dur_aud = pad_trunc(resamples_channels, self.duration)
        shift_aud = time_shift(dur_aud, self.shift_pct)
        _spectrogram = spectrogram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_spectrogram = spectrogram_augment(_spectrogram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return torch.squeeze(aug_spectrogram), class_id


def get_dataloader(test_folder_number: int) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(SoundDS(test_folder_number, test=False), batch_size=16, shuffle=True,
                                  pin_memory=True, num_workers=1)
    test_dataloader = DataLoader(SoundDS(test_folder_number, test=True), batch_size=16, shuffle=False,
                                 pin_memory=True, num_workers=1)
    return train_dataloader, test_dataloader


# dataset = SoundDS(1, test=False)
# print(len(dataset))
