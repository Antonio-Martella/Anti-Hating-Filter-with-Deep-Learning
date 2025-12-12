import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):

    '''
    Custom Keras metric to compute the F1 score during training and evaluation.
    It internally uses the Precision and Recall metrics, updating their states
    for each batch. The F1 score is computed as the harmonic mean of precision
    and recall. Supports resetting states between epochs.
    '''

    def __init__(self, name='f1', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
