# src/dataset.py

import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path="data/"):
    """
    Load dataset from .npy files.
    
    Args:
        data_path (str): Path to folder containing X.npy and Y.npy.
    
    Returns:
        X (np.ndarray): Images, normalized (0–1), shape (N, 64, 64, 1).
        y (np.ndarray): Labels as integers, shape (N,).
    """
    X = np.load(data_path + "X.npy")   # shape: (2062, 64, 64)
    y = np.load(data_path + "Y.npy")   # shape: (2062, 10), one-hot

    # Normalize images (0–255 → 0–1)
    X = X.astype("float32") / 255.0

    # Add channel dimension (grayscale → 1 channel)
    X = np.expand_dims(X, axis=-1)     # shape: (N, 64, 64, 1)

    # Convert one-hot labels to integers
    y = np.argmax(y, axis=1)           # shape: (N,)

    return X, y


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        X (np.ndarray): Input images.
        y (np.ndarray): Labels.
        test_size (float): Proportion for test split.
        val_size (float): Proportion for validation split.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Split train+val into train and val
    val_ratio = val_size / (1 - test_size)  # adjust ratio for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
