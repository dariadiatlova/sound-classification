import pandas as pd
import re

from os import walk

from src.data import DATA_PATH
from typing import List, Tuple


def _get_directory_data(audio_data_path, folder_names) -> Tuple[List[str], List[int]]:
    """
    Function takes a directory and returns a list of absolute filenames in this directory and related classes.
    :param audio_data_path: absolute path to the directory with all folders.
    :param folder_names: directory name.
    :return: list of filenames and classes
    """
    file_names = []
    classes = []

    for folder in folder_names:
        for dir_path, _, filenames in walk(f"{audio_data_path}/{folder}"):
            # list of absolute file paths
            # files_absolute_path = (f"{dir_path}/" + pd.Series(filenames)).tolist()
            # file_names.extend(files_absolute_path)
            for filename in filenames:
                try:
                    pattern = "\-(\d*)-"
                    match = re.search(pattern, filename, re.IGNORECASE)
                    class_name = match.group(1)
                    classes.append(int(class_name))
                    file_names.append(f"{dir_path}/" + filename)
                except AttributeError:
                    # pass if file in a directory does not match a pattern, means it is not a wav file.
                    pass
    return file_names, classes


def get_audio_data(test_folder_number, audio_data_path=f"{DATA_PATH}",
                   test: bool = False) -> Tuple[List[str], List[int]]:
    """
    Returns 4 lists with filenames for train and test data, and classes for train and test data.
    :param test_folder_number: int, number from range [1, ..., 10] to use for validation
    :param audio_data_path: str, file path to the main folder with folders named by a number in a range from 1 to 10.
    :param test: bool, if true returns smaller dataset.
    :return: tuple of 2 lists: with filenames and with classes.
    """
    folder_names = ["fold" + str(i) for i in range(1, 11)]
    test_folder = [folder_names[test_folder_number - 1]]  # list contains 1 folder name
    folder_names.pop(test_folder_number - 1)  # list contains 9 folder names

    if test:
        file_names, classes = _get_directory_data(audio_data_path, test_folder)
    else:
        file_names, classes = _get_directory_data(audio_data_path, folder_names)

    return file_names, classes
