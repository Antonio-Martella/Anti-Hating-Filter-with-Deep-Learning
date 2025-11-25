import random
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------
# REPRODUCIBILITY 
# ---------------------------------------

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
# LIBRARY 
# ---------------------------------------
from utils import load_dataset, preprocess_text, tokenization_and_pad, split_dataset_binary, split_dataset_hate_type


# LOADING AND PREPROCESSING OF THE TEXT CORPUS
df = load_dataset()

# SPLIT AND SAVE THE DATASETS
train_binary_hate, test_binary_hate, val_binary_hate = split_dataset_binary(df=df, test_size=0.2, val_size=0.2, augmentation=True)

# TEXT PREPROCESSING 
train_binary_hate = preprocess_text(train_binary_hate, verbose=True)
test_binary_hate = preprocess_text(test_binary_hate, verbose=True)
val_binary_hate = preprocess_text(val_binary_hate, verbose=True)

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
  max_len_hate, vocabulary_hate_size, tokenizer_binary_hate = tokenization_and_pad(X_train = X_train_binary_hate,
                                                                                       X_test = X_test_binary_hate,
                                                                                       X_val = X_val_binary_hate,
                                                                                       folder = 'binary_hate')

'''# INSTANTIATE THE MODEL AND HYPERPARAMETERS
model_hate_binary = binary_hate_model(vocabulary_size = vocabulary_hate_size,
                                      max_len = max_len_hate,
                                      dropout = 0.3,
                                      optimizer = tf.keras.optimizers.AdamW(learning_rate = 1e-3),
                                      loss = 'binary_crossentropy',
                                      metrics = ['accuracy',
                                                 tf.keras.metrics.AUC(name = 'auc', multi_label=False),
                                                 tf.keras.metrics.Precision(name = 'precision'),
                                                 tf.keras.metrics.Recall(name = 'recall'),
                                                 F1Score(name='f1')])

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

model_hate_binary.evaluate(padded_test_hate_sequences, y_test_binary_hate)'''