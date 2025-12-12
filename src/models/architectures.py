from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential


def binary_hate_model(vocabulary_size, max_len, dropout, optimizer, loss, metrics,
                      lstm_units=32, embedding_dim=64, dense_units=16):
  
  '''
  Builds and compiles a bidirectional LSTM neural network for binary hate-speech detection.
  The architecture includes an embedding layer, a bidirectional LSTM encoder, batch normalization,
  dropout regularization, and dense classification layers with a final sigmoid output.

  Input parameters:
    vocabulary_size: int 
        Size of the token vocabulary.
    max_len: int 
        Fixed sequence length used as model input.
    dropout: float 
        Dropout rate applied after key layers.
    optimizer: keras.optimizers.Optimizer
        Optimizer instance or identifier used for model training.
    loss: str or keras.losses.Loss
        Loss function used during optimization.
    metrics: list
        List of metrics to track during training and evaluation.
    lstm_units: int 
        Number of units in the LSTM layer.
    embedding_dim: int 
        Dimensionality of the token embeddings.
    dense_units: int 
        Number of units in the intermediate dense layer.

  Returns
  -------
  keras.Model
      Compiled Keras Sequential model ready for training.
  '''

  model = Sequential()
  model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = max_len))

  model.add(Bidirectional(LSTM(lstm_units, return_sequences=False, activation='tanh')))
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

def hate_type_model(vocabulary_size, max_len, dropout, optimizer, loss, metrics, 
                    lstm_units=32, embedding_dim=64, dense_units=16):

  '''
  Builds and compiles a multi-label hate-type classification model based on a
  Bidirectional LSTM architecture. The network includes an embedding layer,
  a BiLSTM encoder, batch normalization, dropout for regularization, and dense
  layers for feature extraction.

  Parameters
  ----------
  vocabulary_size: int
      Size of the tokenizer vocabulary used for the embedding layer.
  max_len: int
      Fixed sequence length of input tokenized texts.
  dropout: float
      Dropout rate applied to recurrent and dense layers.
  optimizer: keras.optimizers.Optimizer
      Optimizer used during training.
  loss: str or keras.losses.Loss
      Loss function for multi-label classification, typically binary cross-entropy.
  metrics: list
      List of metrics to monitor during training.
  lstm_units: int, optional
      Number of units in the Bidirectional LSTM encoder.
  embedding_dim: int, optional
      Output dimensionality of the embedding layer.
  dense_units: int, optional
      Number of neurons in the intermediate dense layer.

  Returns
  -------
  keras.Model
      A compiled Keras model ready for training on multi-label hate-type data.
  '''

  model = Sequential()
  model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = max_len))

  model.add(Bidirectional(LSTM(lstm_units, return_sequences=False, activation='tanh')))
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  model.add(Dense(dense_units, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout))

  model.add(Dense(units = 6, activation = 'sigmoid'))
  model.build(input_shape = (None, max_len))

  model.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics)

  return model