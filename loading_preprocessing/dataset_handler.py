import os
from os import listdir
from os.path import join, isfile

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


class DatasetHandler:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
        housing_dir = os.path.join("datasets", "housing", name_prefix)
        os.makedirs(housing_dir, exist_ok=True)
        path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

        filepaths = []
        m = len(data)
        for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
            part_csv = path_format.format(name_prefix, file_idx)
            filepaths.append(part_csv)
            with open(part_csv, "wt", encoding="utf-8") as f:
                if header is not None:
                    f.write(header)
                    f.write("\n")
                for row_idx in row_indices:
                    f.write(",".join([repr(col) for col in data[row_idx]]))
                    f.write("\n")
        return filepaths

    @staticmethod
    def get_paths_csv_files(path):
        paths = []
        for f in listdir(path):
            relative_path = join(path, f)
            if isfile(relative_path):
                paths.append(relative_path)
        return paths

    @staticmethod
    def csv_reader_dataset(filepaths, preprocess, repeat=1, n_readers=5,
                           n_read_threads=None, shuffle_buffer_size=10000,
                           n_parse_threads=5, batch_size=32):
        dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
        dataset = dataset.interleave(
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
            cycle_length=n_readers, num_parallel_calls=n_read_threads)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)

    @staticmethod
    def save_train_test_valid_to_multiple_csv_files(housing, train_data, valid_data, test_data):

        header_cols = housing.feature_names + ["MedianHouseValue"]
        header = ",".join(header_cols)
        train_filepaths = DatasetHandler.save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
        valid_filepaths = DatasetHandler.save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
        test_filepaths = DatasetHandler.save_to_multiple_csv_files(test_data, "test", header, n_parts=10)
        return train_filepaths, valid_filepaths, test_filepaths
