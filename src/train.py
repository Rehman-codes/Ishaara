# src/train.py

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.dataset import load_data, split_data
from src.model import build_cnn


def train_model(
    data_path="data/",
    batch_size=32,
    learning_rate=0.001,
    epochs=50,
    dropout_rate=0.5,
    l2_lambda=0.001,
    patience=5,
    model_save_path="experiments/run1/model.h5"
):
    """
    Train CNN on sign language digit dataset.

    Args:
        data_path (str): Path to dataset.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Maximum number of training epochs.
        dropout_rate (float): Dropout rate.
        l2_lambda (float): L2 regularization factor.
        patience (int): Early stopping patience.
        model_save_path (str): Where to save best model.
    
    Returns:
        model: Trained Keras model.
        history: Training history object.
        (X_train, X_val, X_test, y_train, y_val, y_test): Data splits.
    """

    # Load and split dataset
    X, y = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Build model
    model = build_cnn(input_shape=X_train.shape[1:], 
                      num_classes=10, 
                      dropout_rate=dropout_rate, 
                      l2_lambda=l2_lambda)

    # Compile model
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

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return model, history, (X_train, X_val, X_test, y_train, y_val, y_test)
