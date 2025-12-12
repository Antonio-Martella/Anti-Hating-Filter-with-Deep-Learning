import numpy as np
from sklearn.utils import class_weight


def class_weights_hate(y_train):

  '''
    Computes balanced class weights for a binary or multiclass hate-speech classification task.
    
    Input:
        y_train: array-like of shape (n_samples,)
            Training labels used to estimate the frequency of each class.
    
    Output:
        dict
            A mapping {class_index: weight} where each weight compensates 
            for class imbalance according to scikit-learn’s "balanced" strategy.
  '''

  class_weights_hate = class_weight.compute_class_weight(class_weight = 'balanced',
                                                         classes = np.unique(y_train),
                                                         y = y_train)

  class_weights_hate = dict(enumerate(class_weights_hate))

  return class_weights_hate


# -----------------------------------


def compute_class_weights(y_train):

  '''
  Compute normalized class weights for multilabel classification.

  Parameters
  ----------
  y_train : ndarray of shape (n_samples, n_classes)
      Binary matrix containing the training labels.

  Returns
  -------
  weights : ndarray of shape (n_classes,)
      Class weights inversely proportional to class frequencies, normalized so that
      their average equals 1. These weights can be used to mitigate class imbalance
      during training.
  '''

  class_counts = np.sum(y_train, axis=0)     
  class_freq = class_counts / y_train.shape[0]

  weights = 1.0 / class_freq                 
  weights = weights / np.sum(weights) * len(weights)

  return weights