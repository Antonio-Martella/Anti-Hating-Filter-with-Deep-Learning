import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential


def callback_binary_hate():

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',  # o 'val_auc' ecc.
                                           factor = 0.7,          # riduci LR del 30%
                                           patience = 2,          # aspetta 2 epoche senza miglioramento
                                           min_lr = 1e-6,         # non andare sotto questo valore
                                           verbose = 0)           # stampa i cambiamenti

  early_stop = EarlyStopping(monitor = 'val_loss',         # metrica da monitorare
                             patience = 7,                 # quante epoche aspettare prima di fermarsi
                             restore_best_weights = True,  # ripristina i pesi migliori
                             verbose = 1)

  return early_stop, reduce_learning_rate


#------------------------------------------------------------------


def class_weights_hate(y_train):

  class_weights_hate = class_weight.compute_class_weight(class_weight = 'balanced',
                                                         classes = np.unique(y_train),
                                                         y = y_train)

  class_weights_hate = dict(enumerate(class_weights_hate))

  return class_weights_hate


#------------------------------------------------------------------


def binary_hate_model(vocabulary_size, max_len, dropout, optimizer, loss, metrics):

  model = Sequential()
  model.add(Embedding(input_dim = vocabulary_size, output_dim = 128, input_length = max_len))

  #
  model.add(Bidirectional(LSTM(32, return_sequences=False, activation='tanh')))
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  #
  model.add(Dense(16, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  model.add(Dense(units = 1, activation = 'sigmoid'))
  model.build(input_shape = (None, max_len))

  model.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics)

  return model