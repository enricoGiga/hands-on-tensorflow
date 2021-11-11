import os

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from loading_preprocessing.dataset_handler import DatasetHandler

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "data"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

os.makedirs(IMAGES_PATH, exist_ok=True)
np.random.seed(42)

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]

# train_filepaths, valid_filepaths, test_filepaths = DatasetHandler.save_train_test_valid_to_multiple_csv_files(housing, train_data, valid_data, test_data)

path_to_train = os.path.join("datasets", "housing", "train")
path_to_valid = os.path.join("datasets", "housing", "valid")
path_to_test = os.path.join("datasets", "housing", "test")

train_filepaths = DatasetHandler.get_paths_csv_files(path_to_train)
valid_filepaths = DatasetHandler.get_paths_csv_files(path_to_valid)
test_filepaths = DatasetHandler.get_paths_csv_files(path_to_test)

n_inputs = X_train.shape[-1]

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_


def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y


preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')
tf.random.set_seed(42)

train_set = DatasetHandler.csv_reader_dataset(train_filepaths, preprocess, repeat=None)
valid_set = DatasetHandler.csv_reader_dataset(valid_filepaths, preprocess)
test_set = DatasetHandler.csv_reader_dataset(test_filepaths, preprocess)

tf.random.set_seed(42)

train_set1 = DatasetHandler.csv_reader_dataset(train_filepaths, preprocess, batch_size=2)
for X_batch, y_batch in train_set1.take(2):
    print("X =", X_batch)
    print("y =", y_batch)
    print()

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3))

batch_size = 32
#model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10, validation_data=valid_set)

model.fit((X_train - X_mean) / X_std, y_train, epochs=10, validation_data=(X_valid, y_valid))
