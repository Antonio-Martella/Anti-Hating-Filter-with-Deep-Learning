import optuna
import tensorflow as tf

from models import binary_hate_model, callback_binary_hate
from utils import F1Score
from training import (
    padded_train_hate_sequences,
    y_train_binary_hate,
    max_len_hate,
    vocabulary_hate_size
)


def objective(trial):

    # ---- HYPERPARAMETER SEARCH SPACE ----
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lstm_units = trial.suggest_categorical("lstm_units", [16, 32, 64])
    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # ---- BUILD MODEL ----
    model = binary_hate_model(
        vocabulary_size=vocabulary_hate_size,
        max_len=max_len_hate,
        dropout=dropout,
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(name="f1")
        ],
        lstm_units=lstm_units,
        embedding_dim=embedding_dim
    )

    # ---- YOUR CALLBACKS ----
    early_stop, checkpoint, reduce_lr = callback_binary_hate()

    # ---- TRAIN ----
    history = model.fit(
        padded_train_hate_sequences,
        y_train_binary_hate,
        epochs=30,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=0
    )

    # ---- RETURN BEST VAL_F1 ----
    val_f1 = max(history.history["val_f1"])
    return float(val_f1)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print("MIGLIORI IPERPARAMETRI TROVATI:")
    print(study.best_trial.params)
