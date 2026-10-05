"""Fashion-MNIST dataset loader."""

import os
import gzip
import struct
import urllib.request
from typing import Optional, Tuple

import numpy as np

from dptiny.backend import xp


CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "dptiny",
    "fashion_mnist",
)

_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "zalandoresearch/fashion-mnist/master/data/fashion/"
)

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_file(filename: str, cache_dir: str) -> str:
    """Download a Fashion-MNIST file if it is not already cached."""
    os.makedirs(cache_dir, exist_ok=True)

    file_path = os.path.join(cache_dir, filename)

    # Reuse an existing cached file.
    if os.path.exists(file_path):
        return file_path

    url = _BASE_URL + filename

    # Download to a temporary file first.
    temp_path = file_path + ".tmp"

    try:
        urllib.request.urlretrieve(url, temp_path)

        # Only move it to the final cache location after
        # the download finishes successfully.
        os.replace(temp_path, file_path)

    except Exception:
        # Never leave a partially downloaded file behind.
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return file_path


def _read_idx_gz(file_path: str) -> np.ndarray:
    """Read a gzip-compressed uint8 IDX file."""

    with gzip.open(file_path, "rb") as f:
        data = f.read()

    # An IDX header must contain at least 4 bytes.
    if len(data) < 4:
        raise ValueError("Invalid IDX file: header is too short.")

    # First two bytes must be zero.
    if data[0] != 0 or data[1] != 0:
        raise ValueError("Invalid IDX file header.")

    # 0x08 means unsigned byte (uint8).
    dtype_code = data[2]
    if dtype_code != 0x08:
        raise ValueError("IDX file is not uint8.")

    # Fourth byte tells us the number of dimensions.
    ndim = data[3]

    if ndim == 0:
        raise ValueError("Invalid IDX file: zero dimensions.")

    header_size = 4 + 4 * ndim

    if len(data) < header_size:
        raise ValueError(
            "Invalid IDX file: incomplete dimension header."
        )

    # Read dimension sizes as big-endian unsigned integers.
    shape = struct.unpack(
        ">" + "I" * ndim,
        data[4:header_size],
    )

    expected_size = int(np.prod(shape))
    actual_size = len(data) - header_size

    if actual_size != expected_size:
        raise ValueError(
            "IDX file size does not match the size declared in its header."
        )

    array = np.frombuffer(
        data,
        dtype=np.uint8,
        offset=header_size,
    )

    return array.reshape(shape)


def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    """Load the Fashion-MNIST dataset.

    Args:
        normalize: If True, scale pixel values to [0, 1].
        flatten: If True, return images as (N, 784) vectors.
        data_home: Directory used to cache the downloaded files.

    Returns:
        (X_train, X_test, y_train, y_test) arrays.
    """
    cache_dir = data_home if data_home is not None else _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Download the four dataset files.
    paths = {
        key: _download_file(filename, cache_dir)
        for key, filename in _FILES.items()
    }

    # Parse the IDX files.
    X_train = _read_idx_gz(paths["train_images"])
    y_train = _read_idx_gz(paths["train_labels"])
    X_test = _read_idx_gz(paths["test_images"])
    y_test = _read_idx_gz(paths["test_labels"])

    # Convert images to float32 and labels to int32.
    X_train = X_train.astype(xp.float32)
    X_test = X_test.astype(xp.float32)
    y_train = y_train.astype(xp.int32)
    y_test = y_test.astype(xp.int32)

    # Normalize pixels from [0, 255] to [0, 1].
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    # Match the interface of get_mnist().
    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test
