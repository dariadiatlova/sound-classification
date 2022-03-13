import torch

from typing import Tuple

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from src.preprocessing.audio_metadata import get_audio_data
from src.preprocessing.audio_util_load import (resample, load_wav_file, resample_channel, pad_trunc, time_shift,
                                               spectrogram, spectrogram_augment)


class SoundDS(Dataset):
    def __init__(self, test_fold_number, dataset_config, test: bool = False):
        if test:
            self.audio_files, self.classes = get_audio_data(test_fold_number, test=test)
        else:
            self.audio_files, self.classes = get_audio_data(test_fold_number, test=test)
        assert len(self.audio_files) == len(self.classes), "Number of classes does not match a number of filenames."
        self.duration: int = dataset_config["duration"]
        self.sr: int = dataset_config["sr"]
        self.channel: int = dataset_config["n_audio_channels"]
        self.shift_pct: float = dataset_config["shift_pct"]
        self.n_mels: int = dataset_config["n_mels"]
        self.n_ffts: int = dataset_config["n_ffts"]

    def __len__(self):
        # return 8
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
        _spectrogram = spectrogram(shift_aud, n_mels=self.n_mels, n_fft=self.n_ffts, hop_len=None)
        aug_spectrogram = spectrogram_augment(_spectrogram)

        return torch.squeeze(aug_spectrogram), class_id


def get_dataloader(test_folder_number: int, config_path: str = "src/config.yaml") -> Tuple[DataLoader, DataLoader]:
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    dataset_config = config["dataset"]
    train_dataloader = DataLoader(SoundDS(test_folder_number, dataset_config, test=False),
                                  batch_size=dataset_config["batch_size"],
                                  shuffle=True, pin_memory=True, num_workers=dataset_config["num_workers"])

    test_dataloader = DataLoader(SoundDS(test_folder_number, dataset_config, test=True),
                                 batch_size=dataset_config["batch_size"],
                                 shuffle=False, pin_memory=True, num_workers=dataset_config["num_workers"])
    return train_dataloader, test_dataloader


# config = OmegaConf.load("config.yaml")
# config = OmegaConf.to_container(config, resolve=True)
# dataset_config = config["dataset"]
# dataset = SoundDS(9, dataset_config, test=False)
# print(len(dataset))
