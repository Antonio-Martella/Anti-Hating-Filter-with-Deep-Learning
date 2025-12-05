import numpy as np
import os
from sklearn.metrics import precision_recall_curve
from utils.param_utils import save_param


def f1_score_optimization_thresholds(y_true, y_pred, labels, folder=None):
  optimal_thresholds = {}

  for i, label in enumerate(labels):
    precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    f1_scores[np.isnan(f1_scores)] = 0
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    optimal_thresholds[label] = optimal_threshold

    path = f"models/{folder}"
    os.makedirs(path, exist_ok=True)
    save_param(f"{path}/best_thresholds.json",
               f"best_thresholds_{label}",
               float(optimal_threshold))

  return optimal_thresholds



def f1_score_optimization(y_true, y_pred, folder=None):

    '''
    Computes the optimal classification threshold that maximizes the F1-score 
    based on precision-recall curve and saves it for later use.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1) of the dataset.
    y_pred : array-like
        Predicted probabilities (floats in [0,1]) from the model.
    folder : str, optional
        Subfolder where the threshold parameter JSON will be saved (e.g., 'binary_hate').

    Returns
    -------
    optimal_threshold : float
        The threshold value that maximizes the F1-score.
    
    Output
    ------
    Saves a JSON file "models/{folder}/param_model_{folder}.json" containing the best threshold.
    '''

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    f1_scores[np.isnan(f1_scores)] = 0

    optimal_threshold = thresholds[np.argmax(f1_scores)]
    save_param(f"models/{folder}/param_model_{folder}.json",
               "best_threshold",
               float(optimal_threshold))

    return optimal_threshold
