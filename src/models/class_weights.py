import numpy as np
from sklearn.utils import class_weight


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