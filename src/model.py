# src/model.py

from tensorflow.keras import models, layers, regularizers

def build_cnn(input_shape=(64, 64, 3), num_classes=10, dropout_rate=0.5, l2_lambda=0.001):
    """
    Build CNN model for sign language digit recognition.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(l2_lambda),
                      input_shape=input_shape),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(128, (3,3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_rate),

        layers.Flatten(),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model
