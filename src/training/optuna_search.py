import optuna
import tensorflow as tf
import numpy as np
import os
import math
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models import class_weights_hate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ────────────────────────────────────────────
# IMPORT DAL TUO PROGETTO
# ────────────────────────────────────────────

from utils import load_dataset, preprocess_text, tokenization_and_pad, split_dataset_binary, F1Score

from models import binary_hate_model



# ---------------------------------------
# REPRODUCIBILITY 
# ---------------------------------------

SEED = 42

'''os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = '1'
os.environ["TF_CUDNN_DETERMINISTIC"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["TF_NUM_INTRAOP_THREADS"] = '1'
os.environ["TF_NUM_INTEROP_THREADS"] = '1'''

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------------


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
    lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128])
    dense_units = trial.suggest_categorical("dense_units", [8, 16, 32, 64])
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])

    # ----------------------
    # MODELLO
    # ----------------------
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

    model = binary_hate_model(vocabulary_size = vocabulary_hate_size,
                              max_len = max_len_hate,
                              dropout = dropout,
                              optimizer=optimizer,
                              loss="binary_crossentropy",
                              metrics = [
                                'accuracy',
                                tf.keras.metrics.Precision(name = 'precision'),
                                tf.keras.metrics.Recall(name = 'recall'),
                                F1Score(name='f1')
                                ],
                              lstm_units = lstm_units,
                              embedding_dim = embedding_dim,
                              dense_units = dense_units,
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
        factor=0.8,
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
        validation_split = 0.2,
        #validation_data=(padded_test_hate_sequences, y_test_binary_hate),
        epochs=12,                  
        batch_size=batch_size,
        #class_weight = class_weights_hate(y_train_binary_hate),
        callbacks=[early_stop, reduce_lr]
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
    study.optimize(objective, n_trials=2)

    print("\n───────────────────────────────────────────────")
    print(" BEST HYPERPARAMETERS FOUND ")
    print("───────────────────────────────────────────────")
    print(study.best_params)
    print(f"Best F1 Score: {study.best_value:.4f}")

    # Salva best params
    with open("results/binary_hate/best_hyperparams_binary_hate.json", "w") as f:
      json.dump({**study.best_params, "best_f1": study.best_value}, f, indent=4)



    # -----------------------------------------------------
    # DOPO OPTUNA: VALUTAZIONE SUL TEST SET
    # -----------------------------------------------------
    print("\nTraining final model using best hyperparameters...")

    best_params = study.best_params

    # Ricrea il modello con i best params
    best_model = binary_hate_model(
        vocabulary_size=vocabulary_hate_size,
        max_len=max_len_hate,
        dropout=best_params["dropout"],
        lstm_units=best_params["lstm_units"],
        embedding_dim=best_params["embedding_dim"],
        dense_units=best_params["dense_units"],
        optimizer=tf.keras.optimizers.AdamW(learning_rate=best_params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            F1Score(name="f1"),
        ],
    )

    # Allena su TUTTO il train (train + val)
    best_model.fit(
        padded_train_hate_sequences,
        y_train_binary_hate,
        epochs=20,
        batch_size=best_params["batch_size"],
        #class_weight=class_weights_hate(y_train_binary_hate),
        validation_split=0.0,    # niente validation qui
        verbose=1
    )

    # Predizione sul TEST
    print("\nEvaluating on TEST SET...")
    test_metrics = best_model.evaluate(
        padded_test_hate_sequences,
        y_test_binary_hate,
        verbose=1
    )

    print("\n────────────────────────────")
    print(" RESULTS ON TEST SET")
    print("────────────────────────────")
    for name, value in zip(best_model.metrics_names, test_metrics):
        print(f"{name}: {value:.4f}")

    # Salva anche i risultati sul test
    with open("results/binary_hate/test_results.json", "w") as f:
        json.dump({name: float(value) for name, value in zip(best_model.metrics_names, test_metrics)}, f, indent=4)


