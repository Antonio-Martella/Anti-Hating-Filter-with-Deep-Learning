import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, BatchNormalization, GlobalMaxPooling1D, Layer
from tensorflow.keras.models import Sequential


# ------------------------------
# ---------- CALLBACK ----------
# ------------------------------

def callback_binary_hate():

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',  
                                           factor = 0.75,          
                                           patience = 2,         
                                           min_lr = 1e-6,        
                                           verbose = 0)           

  early_stop = EarlyStopping(monitor = 'val_loss',       
                             patience = 10,                 
                             restore_best_weights = True,
                             verbose = 1)

  return early_stop, reduce_learning_rate

# ------------------------------

def callback_hate_type():

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',   
                                           factor = 0.75,           
                                           patience = 2,            
                                           min_lr = 1e-6,           
                                           verbose = 0)            

  early_stop = EarlyStopping(monitor = 'val_loss',         
                             patience = 10,               
                             restore_best_weights = True,  
                             verbose = 1)

  return early_stop, reduce_learning_rate


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
  model.add(Embedding(input_dim = vocabulary_size, 
                      output_dim = 128, 
                      input_length = max_len))

  model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh')))
  model.add(AttentionLayer())
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  model.add(Dense(32, activation='relu'))
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
                      output_dim = 256, 
                      input_length = max_len))

  model.add(Bidirectional(LSTM(units = 256, return_sequences=True, activation = 'tanh')))
  model.add(AttentionLayer())
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  model.add(Dense(units = 128, activation = 'relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  model.add(Dense(units = 6, activation = 'sigmoid'))
  model.build(input_shape = (None, max_len))

  model.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics)

  return model

# ----------------------------


class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.reduce_sum(output, axis=1)