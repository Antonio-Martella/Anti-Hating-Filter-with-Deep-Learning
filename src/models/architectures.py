from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from .attention_layer import AttentionLayer


def binary_hate_model(vocabulary_size, max_len, dropout, optimizer, loss, metrics,
                      lstm_units=32, embedding_dim=64, dense_units=16):

  model = Sequential()
  model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = max_len))

  model.add(Bidirectional(LSTM(lstm_units, return_sequences=False, activation='tanh')))
  #model.add(AttentionLayer())
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  #
  model.add(Dense(dense_units, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout))
  
  #
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