# main.py

from src.train import train_model
from src.evaluate import plot_training_curves, evaluate_model

def main():
    # Train model
    model, history, (X_train, X_val, X_test, y_train, y_val, y_test) = train_model(
        data_path="data/",
        batch_size=32,
        learning_rate=0.001,
        epochs=30,
        dropout_rate=0.5,
        l2_lambda=0.001,
        patience=5,
        model_save_path="experiments/run1/model.h5"
    )

    # Plot training curves
    plot_training_curves(history)

    # Evaluate model on test set
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
