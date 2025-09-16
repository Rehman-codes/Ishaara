# main.py

import os, json, csv
from src.train import train_model
from src.evaluate import plot_training_curves, evaluate_model
from src.utils import Logger
import sys

def save_history(history, save_dir):
    """Save training history as JSON and CSV."""
    os.makedirs(save_dir, exist_ok=True)
    # JSON
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history.history, f, indent=4)
    # CSV
    with open(os.path.join(save_dir, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.history.keys())
        writer.writerows(zip(*history.history.values()))

def main():
    run_dir = "experiments/run1"   # change per run
    os.makedirs(run_dir, exist_ok=True)

    # Redirect terminal output to log file
    sys.stdout = Logger(os.path.join(run_dir, "run.log"))

    # Train
    model, history, (train_gen, val_gen) = train_model(
        data_path="data/",
        batch_size=32,
        learning_rate=0.001,
        epochs=30,
        dropout_rate=0.3,
        l2_lambda=0.0001,
        patience=5,
        model_save_path=f"{run_dir}/model.h5"
    )

    # Save history
    save_history(history, run_dir)

    # Save plots
    plot_training_curves(history, save_dir=run_dir)

    # Full evaluation (confusion matrix, ROC, classification report)
    evaluate_model(model, val_gen, save_dir=run_dir)

if __name__ == "__main__":
    main()
