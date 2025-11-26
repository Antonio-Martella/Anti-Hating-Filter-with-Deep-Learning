import optuna
import tensorflow as tf
import numpy as np
import os
import math
import sys
import json

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ────────────────────────────────────────────
# IMPORT DAL TUO PROGETTO
# ────────────────────────────────────────────

from utils import load_dataset, preprocess_text, tokenization_and_pad, split_dataset_binary, F1Score

from models import binary_hate_model


# -----------------------------------------------------
# CARICA I DATI - QUESTO È IDENTICO AL TUO train_binary
# -----------------------------------------------------
print("Loading dataset...")
df = load_dataset()

train_binary_hate, test_binary_hate = split_dataset_binary(df = df, 
                                                           test_size = 0.2, 
                                                           augmentation = False)
# TEXT PREPROCESSING 
train_binary_hate = preprocess_text(train_binary_hate, verbose=True)
test_binary_hate = preprocess_text(test_binary_hate, verbose=True)

# TRAINING
X_train_binary_hate = train_binary_hate.comment_text.values
y_train_binary_hate = train_binary_hate.has_hate.values
# TESTING
X_test_binary_hate = test_binary_hate.comment_text.values
y_test_binary_hate = test_binary_hate.has_hate.values

# TOKENIATION AND PUDDING
padded_train_hate_sequences, padded_test_hate_sequences, max_len_hate, \
  vocabulary_hate_size, tokenizer_binary_hate = tokenization_and_pad(X_train = X_train_binary_hate,
                                                                     X_test = X_test_binary_hate,
                                                                     folder = 'binary_hate')

# tokenizzazione & padding
#tokenizer, padded_sequences, vocabulary_size, max_len = tokenization_and_pad(
#    df["processed"].tolist()
#)

# target binario (0-1)
#y = df["label_binary"].values

# split train/valid
#X_train, X_val, y_train, y_val = split_dataset_binary(padded_sequences, y, test_size=0.2)


# -----------------------------------------------------
# FUNZIONE DA OTTIMIZZARE
# -----------------------------------------------------
def objective(trial):
    # ----------------------
    # SAMPLING IPERPARAMETRI
    # ----------------------
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lstm_units = trial.suggest_categorical("lstm_units", [16, 32, 64, 128])
    dense_units = trial.suggest_categorical("dense_units", [8, 16, 32, 64])
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    # ----------------------
    # MODELLO
    # ----------------------
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

    # Costruisci modello con iperparametri
    model = binary_hate_model(
        vocabulary_size=vocabulary_hate_size,
        max_len=max_len_hate,
        dropout=dropout,
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(name="f1"),
        ],
        lstm_units=lstm_units,        # AGGIUNTA IMPORTANTE
        embedding_dim = embedding_dim,
        dense_units=dense_units       # AGGIUNTA IMPORTANTE
    )

    # ----------------------
    # CALLBACKS
    # ----------------------
    early_stop = EarlyStopping(
        monitor="val_f1",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_f1",
        factor=0.7,
        patience=2,
        min_lr=1e-6,
        mode="max",
        verbose=0
    )

    # ----------------------
    # TRAINING BREVE (IMPORTANTE)
    # ----------------------
    history = model.fit(
        padded_train_hate_sequences,
        y_train_binary_hate,
        validation_data=(padded_test_hate_sequences, y_test_binary_hate),
        epochs=12,                   # breve training — abbastanza per capire i trends
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stop, reduce_lr],
    )

    # ----------------------
    # OBIETTIVO DI OPTUNA
    # ----------------------
    val_f1 = max(history.history["val_f1"])

    return val_f1


# -----------------------------------------------------
# LANCIO DI OPTUNA
# -----------------------------------------------------
if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="hate_binary_opt"
    )

    print("Starting hyperparameter search...")
    study.optimize(objective, n_trials=5)

    print("\n───────────────────────────────────────────────")
    print(" BEST HYPERPARAMETERS FOUND ")
    print("───────────────────────────────────────────────")
    print(study.best_params)
    print(f"Best F1 Score: {study.best_value:.4f}")

    # Salva best params
    with open("results/binary_hate/best_hyperparams_binary_hate.json", "w") as f:
      json.dump({**study.best_params, "best_f1": study.best_value}, f, indent=4)

