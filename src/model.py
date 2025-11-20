import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, BatchNormalization, GlobalMaxPooling1D, Layer
from tensorflow.keras.models import Sequential

from layers.attention import AttentionLayer


# ------------------------------
# ---------- CALLBACK ----------
# ------------------------------

def callback_binary_hate():

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',  
                                           factor = 0.5,          
                                           patience = 2,         
                                           min_lr = 1e-6,        
                                           verbose = 0)           

  early_stop = EarlyStopping(monitor = 'val_loss',       
                             patience = 5,                 
                             restore_best_weights = True,
                             verbose = 1)

  checkpoint = ModelCheckpoint('/content/Anti-Hating-Filter-with-Deep-Learning/models/binary_hate/model_hate_binary.h5',
                               monitor = 'val_loss',
                               save_best_only = True,
                               save_weights_only = False,
                               verbose = 1)

  return early_stop, checkpoint, reduce_learning_rate

# ------------------------------

def callback_hate_type():

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',   
                                           factor = 0.5,           
                                           patience = 3,            
                                           min_lr = 1e-6,           
                                           verbose = 0)            

  early_stop = EarlyStopping(monitor = 'val_loss',         
                             patience = 7,               
                             restore_best_weights = True,  
                             verbose = 1)

  checkpoint = ModelCheckpoint(f'/content/Anti-Hating-Filter-with-Deep-Learning/models/hate_type/model_hate_type.h5',
                               monitor = 'val_loss',
                               save_best_only = True,
                               save_weights_only = False,
                               verbose = 1)

  return early_stop, checkpoint, reduce_learning_rate


# -----------------------------------
# ---------- CLASS WEIGHTS ----------
# -----------------------------------

def class_weights_hate(y_train):

  class_weights_hate = class_weight.compute_class_weight(class_weight = 'balanced',
                                                         classes = np.unique(y_train),
                                                         y = y_train)

  class_weights_hate = dict(enumerate(class_weights_hate))

  return class_weights_hate

# -----------------------------------

def compute_class_weights(y_train):

  class_counts = np.sum(y_train, axis=0)
  class_freq = class_counts / y_train.shape[0]

  weights = 1.0 / class_freq
  weights = weights / np.sum(weights) * len(weights)

  return weights


# -----------------------------------
# ---------- LOSS FUNCTION ----------
# -----------------------------------

def weighted_binary_crossentropy(weights):
  
  weights = tf.constant(weights, dtype=tf.float32)
    
  def loss(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * weights, axis=-1)
    
  return loss


# ----------------------------
# ---------- MODELS ----------
# ----------------------------

def binary_hate_model(vocabulary_size, max_len, dropout, optimizer, loss, metrics):

  model = Sequential()
  model.add(Embedding(input_dim = vocabulary_size, output_dim = 64, input_length = max_len))

  #
  model.add(Bidirectional(LSTM(32, return_sequences=True, activation='tanh')))
  model.add(AttentionLayer())
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

# ----------------------------

def hate_type_model(vocabulary_size, max_len, dropout, optimizer, loss, metrics):

  model = Sequential()
  model.add(Embedding(input_dim = vocabulary_size, 
                      output_dim = 128, 
                      input_length = max_len))

  model.add(Bidirectional(LSTM(units = 64, return_sequences=True, activation = 'tanh')))
  model.add(AttentionLayer())
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  model.add(Dense(units = 32, activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  model.add(Dense(units = 6, activation = 'sigmoid'))
  model.build(input_shape = (None, max_len))

  model.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics)

  return model
