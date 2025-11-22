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
from data_utils import load_dataset, preprocess_text, tokenization_and_pudding, CSVLoggerCustom, split_dataset_binary, split_dataset_hate_type
from model import binary_hate_model, callback_binary_hate, class_weights_hate, compute_class_weights, weighted_binary_crossentropy, \
                  hate_type_model, callback_hate_type
from evaluate import evaluation_class, evaluate_model


# LOADING AND PREPROCESSING OF THE TEXT CORPUS
df = load_dataset()

# -------------------------------------------------------------------
# ----- FIRST MODEL, BINARY CLASSIFICATION, HATING OR NOT HATING ----
# -------------------------------------------------------------------
# SPLIT AND SAVE THE DATASETS
train_binary_hate, test_binary_hate, val_binary_hate = split_dataset_binary(df=df, test_size=0.2, val_size=0.2, augmentation=True)

# TEXT PREPROCESSING 
train_binary_hate = preprocess_text(train_binary_hate)
test_binary_hate = preprocess_text(test_binary_hate)
val_binary_hate = preprocess_text(val_binary_hate)

# TRAINING
X_train_binary_hate = train_binary_hate.comment_text.values
y_train_binary_hate = train_binary_hate.has_hate.values
# TESTING
X_test_binary_hate = test_binary_hate.comment_text.values
y_test_binary_hate = test_binary_hate.has_hate.values
# VALIDATION
X_val_binary_hate = val_binary_hate.comment_text.values
y_val_binary_hate = val_binary_hate.has_hate.values

# TOKENIATION AND PUDDING
padded_train_hate_sequences, padded_test_hate_sequences, padded_val_hate_sequences, \
  max_len_hate, vocabulary_hate_size, tokenizer_binary_hate = tokenization_and_pudding(X_train = X_train_binary_hate,
                                                                                       X_test = X_test_binary_hate,
                                                                                       X_val = X_val_binary_hate,
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
csv_logger_binary_hate = CSVLoggerCustom('results/binary_hate/log_training_model_binary_hate.csv', verbose = True)

# FIT THE MODEL
history_hate_binary = model_hate_binary.fit(padded_train_hate_sequences,
                                            y_train_binary_hate,
                                            epochs = 100,
                                            validation_data=(padded_val_hate_sequences, y_val_binary_hate),
                                            batch_size = 256,
                                            class_weight = class_weights_hate(y_train_binary_hate),
                                            callbacks = [callback_binary_hate(), csv_logger_binary_hate])

# COPY WEIGHTS TO /models (to be added)
model_hate_binary.save('/content/drive/MyDrive/Colab Notebooks/Progetto GitHub/DL GitHub/model_hate_binary.h5')

evaluate_model(model_hate_binary, 
               padded_test_hate_sequences, 
               y_test_binary_hate, 
               folder='binary_hate')

model_hate_binary.evaluate(padded_test_hate_sequences, y_test_binary_hate)


# --------------------------------------------------------------
# ----- SECOND MODEL, MULTILABEL CLASSIFICATION, TYPE HATE -----
# --------------------------------------------------------------
# SPLIT AND SAVE THE DATASETS
train_hate_type, test_hate_type, val_hate_type = split_dataset_hate_type(df=df, test_size=0.2, val_size=0.2)

# TEXT PREPROCESSING 
train_hate_type = preprocess_text(train_hate_type)
test_hate_type = preprocess_text(test_hate_type)
val_hate_type = preprocess_text(val_hate_type)

# TRAINING
X_train_hate_type = train_hate_type.comment_text.values
y_train_hate_type = train_hate_type.loc[:, 'toxic':'identity_hate']
# TESTING
X_test_hate_type = test_hate_type.comment_text.values
y_test_hate_type = test_hate_type.loc[:, 'toxic':'identity_hate']
# VALIDATION
X_val_hate_type = val_hate_type.comment_text.values
y_val_hate_type = val_hate_type.loc[:, 'toxic':'identity_hate']


# TOKENIATION AND PUDDING
padded_train_hate_type_sequences, padded_test_hate_type_sequences, padded_val_hate_type_sequences,\
 max_len_hate_type, vocabulary_hate_type_size, tokenizer_hate_type = tokenization_and_pudding(X_train = X_train_hate_type, 
                                                                                              X_test = X_test_hate_type,
                                                                                              X_val = X_val_hate_type,
                                                                                              folder = 'hate_type')

# CALCULATE THE WEIGHTS OF THE CLASSES
weights_tensor = tf.constant(compute_class_weights(y_train_hate_type), dtype=tf.float32)
np.save('results/hate_type/weights_tensor.npy', weights_tensor.numpy())

model_hate_type = hate_type_model(vocabulary_size = vocabulary_hate_type_size,
                                  max_len = max_len_hate_type,
                                  dropout = 0.3,
                                  optimizer = tf.keras.optimizers.AdamW(learning_rate = 1e-3),
                                  loss = weighted_binary_crossentropy(weights_tensor),
                                  metrics = ['accuracy',
                                             tf.keras.metrics.AUC(name = 'auc', multi_label=True),
                                             tf.keras.metrics.Precision(name = 'precision'),
                                             tf.keras.metrics.Recall(name = 'recall')])

csv_logger_hate_type = CSVLoggerCustom('results/hate_type/log_training_model_hate_type.csv', verbose = True)

history_hate_type = model_hate_type.fit(padded_train_hate_type_sequences,
                                        y_train_hate_type,
                                        epochs = 100,
                                        validation_data=(padded_val_hate_type_sequences, y_val_hate_type),
                                        batch_size = 64,
                                        callbacks = [callback_hate_type(), csv_logger_hate_type])

model_hate_type.save('/content/drive/MyDrive/Colab Notebooks/Progetto GitHub/DL GitHub/model_hate_type.h5')