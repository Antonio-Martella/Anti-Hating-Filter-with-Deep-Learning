import optuna
import numpy as np
import os
import math
import sys
import json
import random

import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils import load_dataset, preprocess_text, tokenization_and_pad, split_dataset_binary, F1Score
from models import binary_hate_model, class_weights_hate



# ---------------------------------------
# REPRODUCIBILITY 
# ---------------------------------------

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
#os.environ["TF_DETERMINISTIC_OPS"] = '1'
#os.environ["TF_CUDNN_DETERMINISTIC"] = '1'
#os.environ["OMP_NUM_THREADS"] = '1'
#os.environ["TF_NUM_INTRAOP_THREADS"] = '1'
#os.environ["TF_NUM_INTEROP_THREADS"] = '1'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------------

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

def objective(trial):

    # HYPERPARAMETERS
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    dense_units = trial.suggest_categorical("dense_units", [8, 16])

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    if lstm_units >= embedding_dim:
        raise optuna.TrialPruned("LSTM units must be < embedding dim")

    # MODEL
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


    # CALLBACKS
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

    
    history = model.fit(
        padded_train_hate_sequences,
        y_train_binary_hate,
        validation_split = 0.2,
        epochs=10,                  
        batch_size=batch_size,
        class_weight = class_weights_hate(y_train_binary_hate),
        callbacks=[early_stop, reduce_lr]
    )

    y_pred_test = model.predict(padded_test_hate_sequences)
    f1_test = f1_score(y_test_binary_hate, y_pred_test>=0.5)
    print(f"Trial {trial.number} — F1 test (not used for tuning): {f1_test:.4f}")

    
    # OPTUNA GOAL
    val_f1 = max(history.history["val_f1"])

    return val_f1


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="hate_binary_opt",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    print("\033[92mStarting hyperparameter search...\033[0m")
    study.optimize(objective, n_trials=15)

    print("\033[92m\n───────────────────────────────────────────────\033[0m")
    print("\033[92m BEST HYPERPARAMETERS FOUND \033[0m")
    print("\033[92m───────────────────────────────────────────────\033[0m")
    print(study.best_params)
    print(f"\033[92mBest F1 Score: {study.best_value:.4f}\033[0m")

    # Save best params
    with open("results/binary_hate/best_hyperparams_binary_hate.json", "w") as f:
      json.dump({**study.best_params, "best_f1": study.best_value}, f, indent=4)


