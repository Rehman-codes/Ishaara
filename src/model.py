# src/model.py

from tensorflow.keras import models, layers, regularizers

def build_cnn(input_shape=(64, 64, 1), num_classes=10, dropout_rate=0.5, l2_lambda=0.001):
    """
    Build a CNN model for sign language digit recognition.
    
    Args:
        input_shape (tuple): Shape of input images (H, W, C).
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        l2_lambda (float): L2 regularization factor.
    
    Returns:
        model (tf.keras.Model): CNN model (uncompiled).
    """
    model = models.Sequential([
        # Conv block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda),
                      input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Conv block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Conv block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
