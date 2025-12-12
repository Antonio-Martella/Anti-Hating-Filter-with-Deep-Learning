import tensorflow as tf


def weighted_binary_crossentropy(weights):

  '''
  This function builds a weighted binary cross-entropy loss.  
  It receives a pair of class weights and returns a loss function that applies them element-wise to the standard binary cross-entropy.  
  Inputs:  
  - weights: a list or array of two float values representing the weights for the negative and positive class.  
  Output:  
  - A callable loss function taking (y_true, y_pred) and returning the weighted binary cross-entropy averaged over the last dimension.
  '''
  
  weights = tf.constant(weights, dtype=tf.float32)
    
  def loss(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * weights, axis=-1)
    
  return loss