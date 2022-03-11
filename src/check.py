import re
import pandas as pd
import torchaudio

from src.data import DATA_PATH
from os import walk, remove
from typing import List


def _walk(audio_data_path, folder_names, check: bool = True) -> List[str]:
    """
    Function takes a directory and returns a list of absolute filenames in this directory and related classes.
    :param audio_data_path: absolute path to the directory with all folders.
    :param folder_names: directory name.
    :return: list of filenames and classes
    """
    file_names = []
    corrupted_files = []

    for folder in folder_names:
        for dir_path, _, filenames in walk(f"{audio_data_path}/{folder}"):
            # list of absolute file paths
            files_absolute_path = (f"{dir_path}/" + pd.Series(filenames)).tolist()
            file_names.extend(files_absolute_path)

    files_were_not_read = 0
    for audio_path in file_names:
        pattern = ".*\.wav"
        is_wav = re.search(pattern, audio_path, re.IGNORECASE)
        if is_wav is not None:
            try:
                signal, sr = torchaudio.load(audio_path)
                if signal is None:
                    corrupted_files.append(audio_path)
            except Exception:
                if check:
                    corrupted_files.append(audio_path)
                    files_were_not_read += 1
                else:
                    remove(audio_path)
        else:
            pass

    print(f"There are in total {len(file_names)} files, couldn't read {files_were_not_read} files.")
    return corrupted_files


_walk(f"{DATA_PATH}", ["fold" + f"{i}" for i in range(11)])
