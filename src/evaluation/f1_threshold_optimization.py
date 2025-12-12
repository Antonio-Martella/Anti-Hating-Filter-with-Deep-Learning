import numpy as np
import os
from sklearn.metrics import precision_recall_curve
from utils.param_utils import save_param


def f1_score_optimization_thresholds(y_true, y_pred, labels, folder=None):

    '''
    Computes the F1-optimal threshold for each label using the precision–recall curve.
    Given true labels, predicted probabilities, and label names, the function identifies
    the threshold that maximizes the F1 score (binary or multilabel) and returns these
    values in a dictionary. If a folder is provided, the thresholds are also saved under
    `models/{folder}`.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (1D for binary, 2D for multilabel).

    y_pred : array-like
        Predicted probabilities with the same shape as `y_true`.

    labels : list of str
        Names of the labels.

    folder : str, optional
        Directory name used to save the computed thresholds.

    Returns
    -------
    dict
        Mapping from each label to its optimal F1-maximizing threshold.
    '''

    optimal_thresholds = {}

    if folder is not None:
        path = f"models/{folder}"
        os.makedirs(path, exist_ok=True)

    # Binary case
    if y_true.ndim == 1:

        label = labels[0]

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        f1_scores[np.isnan(f1_scores)] = 0

        optimal_threshold = float(thresholds[np.argmax(f1_scores)])

        if folder is not None:
          save_param(f"{path}/best_thresholds_{folder}.json",
            f"best_thresholds_{label}",
            float(optimal_threshold))
          
        optimal_thresholds[label] = optimal_threshold
    # Mutlilabel case
    else:
        for i, label in enumerate(labels):
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])

            f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
            f1_scores[np.isnan(f1_scores)] = 0

            optimal_threshold = float(thresholds[np.argmax(f1_scores)])

            if folder is not None:
              save_param(f"{path}/best_thresholds_{folder}.json",
                f"best_thresholds_{label}",
                float(optimal_threshold))

            optimal_thresholds[label] = optimal_threshold

    return optimal_thresholds
