# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import tensorflow as tf
import os


def plot_training_curves(history, save_dir=None):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.legend(), plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend(), plt.title("Loss")

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()


def evaluate_model(model, val_gen, save_dir="experiments/run1"):
    """Evaluate model, save metrics, plots, and reports."""

    os.makedirs(save_dir, exist_ok=True)

    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Save human-readable report
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted"), plt.ylabel("True"), plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # ROC curves (multi-class one-vs-rest)
    try:
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=10)
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(10):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Macro-average AUC
        auc_score = roc_auc_score(y_true_onehot, y_pred_probs, multi_class="ovr")

        plt.figure(figsize=(8,6))
        for i in range(10):
            plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC={roc_auc[i]:.2f})")
        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("False Positive Rate"), plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves (Macro AUC={auc_score:.4f})")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "roc_curves.png"))
        plt.close()

        with open(os.path.join(save_dir, "roc_auc.txt"), "w") as f:
            f.write(f"Macro-average ROC-AUC: {auc_score:.4f}\n")

    except Exception as e:
        print("ROC-AUC error:", e)
