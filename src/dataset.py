# src/dataset.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(
    data_dir="data/",
    img_size=(64, 64),
    batch_size=32,
    val_split=0.2
):
    """
    Load dataset from folder structure using ImageDataGenerator.
    - Assumes data_dir has subfolders '0', '1', ..., '9'.

    Args:
        data_dir (str): Path to dataset root.
        img_size (tuple): Target image size.
        batch_size (int): Batch size for training.
        val_split (float): Fraction for validation split.

    Returns:
        train_gen, val_gen: Data generators.
    """
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=val_split
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="sparse",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen
