import argparse
import json
from typing import Dict

import torch
from omegaconf import OmegaConf

from src.model.ligtning_model import Classifier
from src.preprocessing.audio_util_load import load_wav_file, resample, resample_channel, pad_trunc, time_shift, \
    spectrogram, spectrogram_augment


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-p', '--audio_file_path',
                        help='Path to the wav file to evaluate.',
                        type=str)

    parser.add_argument('-w', '--model_weights_path',
                        help='Path to the .ckpt file to initialize model weights.',
                        type=str)


def feature_extraction(audio_path: str, dataset_config: Dict):
    audio_data = load_wav_file(audio_path)
    resampled_audio = resample(audio_data, dataset_config["sr"])
    resamples_channels = resample_channel(resampled_audio, dataset_config["n_audio_channels"])
    dur_aud = pad_trunc(resamples_channels, dataset_config["duration"])
    shift_aud = time_shift(dur_aud, dataset_config["shift_pct"])
    _spectrogram = spectrogram(shift_aud, dataset_config["n_mels"], dataset_config["n_ffts"], dataset_config["hop_len"])
    aug_spectrogram = spectrogram_augment(_spectrogram)
    return aug_spectrogram


def predict_class_for_one_audio(audio_file_path: str,
                                train_config_path: str,
                                class_names_file_path: str,
                                model_weights_file_path: str) -> int:

    class_names = json.load(open(class_names_file_path))
    config = OmegaConf.load(train_config_path)
    config = OmegaConf.to_container(config, resolve=True)

    train_config = config["train"]
    dataset_config = config["dataset"]

    audio_feature_representation = feature_extraction(audio_file_path, dataset_config)
    model = Classifier.load_from_checkpoint(checkpoint_path=model_weights_file_path, train_config=train_config)
    model.eval()
    log_probs = model(audio_feature_representation)
    # predict class
    predicted_class = class_names[str(int(torch.argmax(log_probs)))]
    return predicted_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    test_sample_path = args.audio_file_path
    model_weights = args.model_weights_path
    predicted_class = predict_class_for_one_audio(test_sample_path,
                                                  train_config_path="src/config.yaml",
                                                  class_names_file_path="src/evaluation/classes_names.json",
                                                  model_weights_file_path=model_weights)
    print(f"Predicted class for the audio: {predicted_class}")
