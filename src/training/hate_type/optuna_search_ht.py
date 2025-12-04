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

from utils import load_dataset, preprocess_text, tokenization_and_pad, split_dataset_hate_type, F1Score
from models import hate_type_model, weighted_binary_crossentropy, class_weights_hate, compute_class_weights

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

#print("Loading dataset...")
df = load_dataset()

train_hate_type, test_hate_type = split_dataset_hate_type(df = df, 
                                                          test_size = 0.2)

# TEXT PREPROCESSING 
train_hate_type = preprocess_text(train_hate_type, verbose=True)
test_hate_type = preprocess_text(test_hate_type, verbose=True)

# TRAINING
X_train_hate_type = train_hate_type.comment_text.values
y_train_hate_type = train_hate_type.loc[:, 'toxic':'identity_hate'].values

# TESTING
X_test_hate_type = test_hate_type.comment_text.values
y_test_hate_type = test_hate_type.loc[:, 'toxic':'identity_hate'].values

# TOKENIATION AND PUDDING
padded_train_hate_sequences, padded_test_hate_sequences, max_len_hate, \
  vocabulary_hate_size, tokenizer_binary_hate = tokenization_and_pad(X_train = X_train_hate_type,
                                                                     X_test = X_test_hate_type,
                                                                     folder = 'hate_type')

weights_tensor = tf.constant(compute_class_weights(y_train_hate_type), dtype=tf.float32)

def objective(trial):

    # HYPERPARAMETERS
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    lstm_units = trial.suggest_categorical("lstm_units", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    dense_units = trial.suggest_categorical("dense_units", [64, 96, 128, 256])

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    if lstm_units >= embedding_dim:
        raise optuna.TrialPruned("LSTM units must be < embedding dim")

    # MODEL
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

    model = hate_type_model(
      vocabulary_size = vocabulary_hate_size,
      max_len = max_len_hate,
      dropout = dropout,
      optimizer=optimizer,
      loss = weighted_binary_crossentropy(weights_tensor),
      metrics = [
        'accuracy',
        tf.keras.metrics.AUC(name = 'auc', multi_label=True),
        tf.keras.metrics.Precision(name = 'precision'),
        tf.keras.metrics.Recall(name = 'recall'),
        F1Score(name='f1')
        ],
      lstm_units = lstm_units,
      embedding_dim = embedding_dim,
      dense_units = dense_units
    )
      
    # CALLBACKS
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        ##mode="max",
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=2,
        min_lr=1e-6,
        #mode="max",
        verbose=0
    )

    
    history = model.fit(
        padded_train_hate_sequences,
        y_train_hate_type,
        validation_split = 0.2,
        epochs=10,                  
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr]
    )

    y_pred_test = model.predict(padded_test_hate_sequences)

    f1_test = f1_score(y_test_hate_type, y_pred_test>=0.5, average='micro')
    print(f"Trial {trial.number} — F1 test (not used for tuning): {f1_test:.4f}")

    
    # OPTUNA GOAL
    #val_f1 = max(history.history["val_f1"])

    val_loss = min(history.history["val_loss"])
    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
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
    with open("results/hate_type/best_hyperparams_hate_type.json", "w") as f:
      json.dump({**study.best_params, "best_f1": study.best_value}, f, indent=4)


