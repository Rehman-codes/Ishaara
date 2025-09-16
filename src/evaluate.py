# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)
import seaborn as sns
import tensorflow as tf


def plot_training_curves(history):
    """Plot training & validation accuracy and loss curves."""
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with accuracy, confusion matrix,
    precision, recall, and ROC-AUC.
    """
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Accuracy, precision, recall, F1
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC-AUC (macro-average for multi-class)
    try:
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=10)
        auc_score = roc_auc_score(y_test_onehot, y_pred_probs, multi_class="ovr")
        print(f"Macro-average ROC-AUC: {auc_score:.4f}")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)
