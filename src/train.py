import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TE_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TE_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns

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
from model import binary_hate_model, callback_binary_hate, class_weights_hate


# PREPROCESSING TESTO
df = load_dataset()
df = preprocess_text(df)

# FIRST MODEL, BINARY CLASSIFICATION, HATING OR NOT HATING
df['has_hate'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].any(axis = 1).astype(int)
x = df.comment_text.values
y_hate = df.loc[:, 'has_hate']

x_train_hate, x_test_hate, y_train_hate, y_test_hate = train_test_split(x, y_hate, test_size = 0.2, 
                                                                        random_state = 1, 
                                                                        stratify = y_hate, 
                                                                        shuffle = True)


padded_train_hate_sequences, padded_test_hate_sequences, \
    max_len_hate, vocabulary_hate_size, _ = tokenization_and_pudding(x_train = x_train_hate,
                                                                  x_test = x_test_hate)

clear_session()
model_hate_binary = binary_hate_model(vocabulary_size = vocabulary_hate_size,
                                      max_len = max_len_hate,
                                      dropout = 0.3,
                                      optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-2),
                                      loss = 'binary_crossentropy',
                                      metrics = ['accuracy',
                                                 tf.keras.metrics.AUC(name = 'auc', multi_label=False),
                                                 tf.keras.metrics.Precision(name = 'precision'),
                                                 tf.keras.metrics.Recall(name = 'recall')])
#model_hate_binary.summary()

csv_logger = CSVLoggerCustom('../results/training_log_model_hate_or_not.csv', verbose = True)

history_hate_binary = model_hate_binary.fit(padded_train_hate_sequences,
                                            y_train_hate,
                                            epochs = 100,
                                            validation_split = 0.2,
                                            batch_size = 256,
                                            class_weight = class_weights_hate(y_test_hate),
                                            callbacks = [callback_binary_hate(), csv_logger])

os.makedirs('models', exist_ok=True) 
model_hate_binary.save('models/hate_filter_model.h5')