# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf

def plot_training_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.legend(), plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend(), plt.title("Loss")
    plt.show()

def evaluate_model(model, val_gen):
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted"), plt.ylabel("True")
    plt.show()

    try:
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=10)
        auc_score = roc_auc_score(y_true_onehot, y_pred_probs, multi_class="ovr")
        print(f"Macro-average ROC-AUC: {auc_score:.4f}")
    except Exception as e:
        print("ROC-AUC error:", e)
