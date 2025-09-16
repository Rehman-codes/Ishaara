# main.py

from src.train import train_model
from src.evaluate import plot_training_curves, evaluate_model

def main():
    model, history, (train_gen, val_gen) = train_model(
        data_path="data/Dataset/",
        batch_size=32,
        learning_rate=0.001,
        epochs=30,
        dropout_rate=0.3,
        l2_lambda=0.0001,
        patience=5,
        model_save_path="experiments/run1/model.h5"
    )

    plot_training_curves(history)
    evaluate_model(model, val_gen)

if __name__ == "__main__":
    main()
