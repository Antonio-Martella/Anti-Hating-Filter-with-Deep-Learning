import random
import numpy as np
import tensorflow as tf
import pandas as pdx
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from sklearn.metrics import precision_recall_curve

from utils import load_dataset, preprocess_text, tokenization_and_pad, split_dataset_hate_type, \
  CSVLoggerCustom, F1Score
from models import hate_type_model, callback_hate_type, compute_class_weights, \
  weighted_binary_crossentropy
from evaluation import evaluate_model, plot_class_distribution


# ---------------------------------------
# REPRODUCIBILITY 
# ---------------------------------------

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = '1'
os.environ["TF_CUDNN_DETERMINISTIC"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["TF_NUM_INTRAOP_THREADS"] = '1'
os.environ["TF_NUM_INTEROP_THREADS"] = '1'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------------

# LOADING AND PREPROCESSING OF THE TEXT CORPUS
df = load_dataset()

# CLASS DISTRIBUTION
#binary_series = (df['sum_injurious'] >= 1).astype(int)
#count = binary_series.value_counts().sort_index()
#plot_class_distribution(count, folder='binary_hate')

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
np.save('results/hate_type/weights_tensor.npy', weights_tensor.numpy())

with open('results/hate_type/best_hyperparams_hate_type.json', "r") as f:
    best_hyperparams = json.load(f)

# INSTANTIATE THE MODEL AND HYPERPARAMETERS
model_hate_type = hate_type_model(
  vocabulary_size = vocabulary_hate_size,
  max_len = max_len_hate,
  dropout = best_hyperparams['dropout'],
  lstm_units = best_hyperparams['lstm_units'],
  embedding_dim = best_hyperparams['embedding_dim'],
  dense_units = best_hyperparams['dense_units'],
  optimizer = tf.keras.optimizers.AdamW(learning_rate = best_hyperparams['learning_rate']),
  loss = weighted_binary_crossentropy(weights_tensor),
  metrics = [
    'accuracy',
     tf.keras.metrics.AUC(name = 'auc', multi_label=True),
     tf.keras.metrics.Precision(name = 'precision'),
     tf.keras.metrics.Recall(name = 'recall'),
     F1Score(name='f1')
     ])


# LOG FILE .csv
csv_logger_hate_type = CSVLoggerCustom('results/hate_type/log_training_model_hate_type.csv', verbose=True)

# FIT THE MODEL
history_hate_binary = model_hate_type.fit(
  padded_train_hate_sequences,
   y_train_hate_type,
   epochs = 100,
   validation_split = 0.2,
   batch_size = best_hyperparams['batch_size'],
   callbacks = [callback_hate_type(), csv_logger_hate_type]
)

# COPY WEIGHTS TO /models (to be added)
#model_hate_type.save('/content/drive/MyDrive/Colab Notebooks/Progetto GitHub/DL GitHub/model_hate_type.keras')
#model_hate_binary.save('models/binary_hate/model_hate_binary.keras')


### -----

y_pred_multi = model_hate_type.predict(padded_test_hate_sequences)
y_true_multi = y_test_hate_type

# Se serve NumPy più tardi:
optimal_thresholds_multilabel = {}
labels = test_hate_type.loc[:, 'toxic':'identity_hate'].columns.tolist()

print("\n\033[92mThresholds ottimali per la classificazione multilabel (massimizzando l'F1-score per ogni classe sull'intero set di test multilabel):\033[0m")

for i, label in enumerate(labels):
    precision_multi, recall_multi, thresholds_multi = precision_recall_curve(y_true_multi[:, i], y_pred_multi[:, i])
    # Escludiamo l'ultimo valore di precision e recall che corrisponde a un threshold non definito
    f1_scores_multi = (2 * precision_multi[:-1] * recall_multi[:-1]) / (precision_multi[:-1] + recall_multi[:-1])
    f1_scores_multi[np.isnan(f1_scores_multi)] = 0 # Handle NaN values

    optimal_threshold = thresholds_multi[np.argmax(f1_scores_multi)]

    optimal_thresholds_multilabel[label] = optimal_threshold
    print(f"\033[92m{label}: {optimal_threshold:.3f}\033[0m")
