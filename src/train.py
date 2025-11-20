# ---------------------------------------
# WARNING
# ---------------------------------------

import os, sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w') 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
sys.stderr = stderr  

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# ---------------------------------------
# REPRODUCIBILITY 
# ---------------------------------------
import random
import numpy as np

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = '0'
os.environ["TF_CUDNN_DETERMINISTIC"] = '0'
os.environ["OMP_NUM_THREADS"] = '0'
os.environ["TF_NUM_INTRAOP_THREADS"] = '0'
os.environ["TF_NUM_INTEROP_THREADS"] = '0'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------
# REPRODUCIBILITY 
# ---------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score, \
                            precision_score, recall_score, f1_score, classification_report, precision_recall_curve

from tensorflow.keras.backend import clear_session


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model, Sequential


# FROM MY FILES
from data_utils import load_dataset, preprocess_text, tokenization_and_pudding, CSVLoggerCustom
from model import binary_hate_model, callback_binary_hate, class_weights_hate, compute_class_weights, weighted_binary_crossentropy, \
                  hate_type_model, callback_hate_type
from evaluate import evaluation_class, evaluate_model



# LOADING AND PREPROCESSING OF THE TEXT CORPUS
df = load_dataset()
df = preprocess_text(df)


# -------------------------------------------------------------------
# ----- FIRST MODEL, BINARY CLASSIFICATION, HATING OR NOT HATING ----
# -------------------------------------------------------------------
df['has_hate'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].any(axis = 1).astype(int)
x = df.comment_text.values
y_hate = df.loc[:, 'has_hate']

# EVALUTATE CLASS DISTRIBUTIONS
class_counts = pd.Series(y_hate).value_counts().sort_index()
evaluation_class(count = class_counts, folder = 'binary_hate')

# SPLIT DATASET
x_train_hate, x_test_hate, \
  y_train_hate, y_test_hate = train_test_split(x, 
                                               y_hate, 
                                               test_size = 0.2, 
                                               random_state = 1, 
                                               stratify = y_hate, 
                                               shuffle = True)

# TOKENIATION AND PUDDING
padded_train_hate_sequences, padded_test_hate_sequences, max_len_hate, vocabulary_hate_size, \
  tokenizer_binary_hate = tokenization_and_pudding(x_train = x_train_hate,
                                                   x_test = x_test_hate,
                                                   folder = 'binary_hate')

# INSTANTIATE THE MODEL AND HYPERPARAMETERS
model_hate_binary = binary_hate_model(vocabulary_size = vocabulary_hate_size,
                                      max_len = max_len_hate,
                                      dropout = 0.3,
                                      optimizer = tf.keras.optimizers.AdamW(learning_rate = 1e-3),
                                      loss = 'binary_crossentropy',
                                      metrics = ['accuracy',
                                                 tf.keras.metrics.AUC(name = 'auc', multi_label=False),
                                                 tf.keras.metrics.Precision(name = 'precision'),
                                                 tf.keras.metrics.Recall(name = 'recall')])

# LOG FILE .csv
csv_logger_binary_hate = CSVLoggerCustom('/content/Anti-Hating-Filter-with-Deep-Learning/results/binary_hate/log_training_model_binary_hate.csv', verbose = True)

# FIT THE MODEL
history_hate_binary = model_hate_binary.fit(padded_train_hate_sequences,
                                            y_train_hate,
                                            epochs = 100,
                                            validation_split = 0.2,
                                            batch_size = 256,
                                            class_weight = class_weights_hate(y_test_hate),
                                            callbacks = [callback_binary_hate(), csv_logger_binary_hate])

# COPY WEIGHTS TO /models (to be added)
model_hate_binary.save('/content/drive/MyDrive/Colab Notebooks/Progetto GitHub/DL GitHub/model_hate_binary.h5')
#model_hate_binary.save('../models/model_hate_binary.h5')

evaluate_model(model_hate_binary, 
               padded_test_hate_sequences, 
               y_test_hate, 
               folder='binary_hate')


model_hate_binary.evaluate(padded_test_hate_sequences,y_test_hate)

y_pred = model_hate_binary.predict(padded_test_hate_sequences)

for i in range(100):
  print(y_pred[i], y_test_hate.iloc[i])

# --------------------------------------------------------------
# ----- SECOND MODEL, MULTILABEL CLASSIFICATION, TYPE HATE -----
# --------------------------------------------------------------
# SELECT COMMENTS WITH AT LEAST ONE TYPE OF HATE
df_hate_type = df[df["has_hate"] == 1]
x_hate_type = df_hate_type.comment_text.values
y_hate_type = df_hate_type.loc[:, 'toxic':'identity_hate']

# EVALUTATE CLASS DISTRIBUTIONS
class_counts = y_hate_type.sum().sort_values(ascending=False)
evaluation_class(count = class_counts, folder = 'hate_type')

# SPLIT DATASET 
x_train_hate_type, x_test_hate_type, \
  y_train_hate_type, y_test_hate_type = train_test_split(x_hate_type,
                                                         y_hate_type, 
                                                         test_size = 0.2, 
                                                         random_state = 1, 
                                                         shuffle = True)

# TOKENIATION AND PUDDING
padded_train_hate_type_sequences, padded_test_hate_type_sequences, max_len_hate_type, vocabulary_hate_type_size, \
  tokenizer_hate_type = tokenization_and_pudding(x_train = x_train_hate_type,
                                                 x_test = x_test_hate_type,
                                                 folder = 'hate_type')

# CALCULATE THE WEIGHTS OF THE CLASSES
weights_tensor = tf.constant(compute_class_weights(y_train_hate_type), dtype=tf.float32)
np.save('/content/Anti-Hating-Filter-with-Deep-Learning/results/hate_type/weights_tensor.npy', weights_tensor.numpy())

model_hate_type = hate_type_model(vocabulary_size = vocabulary_hate_type_size,
                                  max_len = max_len_hate_type,
                                  dropout = 0.3,
                                  optimizer = tf.keras.optimizers.AdamW(learning_rate = 1e-3),
                                  loss = weighted_binary_crossentropy(weights_tensor),
                                  metrics = ['accuracy',
                                             tf.keras.metrics.AUC(name = 'auc', multi_label=True),
                                             tf.keras.metrics.Precision(name = 'precision'),
                                             tf.keras.metrics.Recall(name = 'recall')])

csv_logger_hate_type = CSVLoggerCustom('/content/Anti-Hating-Filter-with-Deep-Learning/results/hate_type/log_training_model_hate_type.csv', verbose = True)

history_hate_type = model_hate_type.fit(padded_train_hate_type_sequences,
                                        y_train_hate_type,
                                        epochs = 100,
                                        validation_split = 0.2,
                                        batch_size = 128,
                                        callbacks = [callback_hate_type(), csv_logger_hate_type])

model_hate_binary.save('/content/drive/MyDrive/Colab Notebooks/Progetto GitHub/DL GitHub/model_hate_type.h5')
#model_hate_type.save('../models/model_hate_type.h5')