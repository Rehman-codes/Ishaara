# src/train.py

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.dataset import get_data_generators
from src.model import build_cnn

def train_model(
    data_path="data/Dataset/",
    batch_size=32,
    learning_rate=0.001,
    epochs=30,
    dropout_rate=0.5,
    l2_lambda=0.001,
    patience=5,
    model_save_path="experiments/run1/model.h5"
):
    # Data
    train_gen, val_gen = get_data_generators(data_path, batch_size=batch_size)

    # Model
    model = build_cnn(input_shape=(64, 64, 3),
                      num_classes=10,
                      dropout_rate=dropout_rate,
                      l2_lambda=l2_lambda)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return model, history, (train_gen, val_gen)
