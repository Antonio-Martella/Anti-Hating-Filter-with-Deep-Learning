import tensorflow as tf


def weighted_binary_crossentropy(weights):
  
  weights = tf.constant(weights, dtype=tf.float32)
    
  def loss(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * weights, axis=-1)
    
  return loss