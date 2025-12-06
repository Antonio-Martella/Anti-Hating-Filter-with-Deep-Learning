import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from .f1_threshold_optimization import f1_score_optimization, f1_score_optimization_thresholds
from .class_distribution import plot_class_distribution


def evaluate_model(model, X_padded, y_true, labels, folder=None):
   
    y_pred = model.predict(X_padded)

    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    is_binary = (y_true.ndim == 1)

    optimal_thresholds = f1_score_optimization_thresholds(
        y_true, y_pred, labels, folder
    )

    metrics_data = []

    if is_binary:
        label = labels[0]
        thr = optimal_thresholds[label]

        y_pred_opt = (y_pred >= thr).astype(int)

        metrics_data.append({
            'Classe': label,
            'Accuracy': accuracy_score(y_true, y_pred_opt),
            'Precision': precision_score(y_true, y_pred_opt, zero_division=0),
            'Recall': recall_score(y_true, y_pred_opt, zero_division=0),
            'F1': f1_score(y_true, y_pred_opt, zero_division=0)
        })

    else:
        for i in range(len(labels)):
          label = labels[i]
          optimal_threshold = optimal_thresholds[label]

          y_pred_opt = (y_pred[:, i] >= optimal_threshold).astype(int)

          acc = accuracy_score(y_true[:, i], y_pred_opt)
          prec = precision_score(y_true[:, i], y_pred_opt, zero_division = 0)
          rec = recall_score(y_true[:, i], y_pred_opt, zero_division = 0)
          f1 = f1_score(y_true[:, i], y_pred_opt, zero_division = 0)
          metrics_data.append({'Classe': label,
                              'Accuracy': acc,
                              'Precision': prec,
                              'Recall': rec,
                              'F1': f1})

    metrics_df = pd.DataFrame(metrics_data)

    if folder is not None:
        os.makedirs(f"results/{folder}", exist_ok=True)
        metrics_df.to_csv(f"results/{folder}/metrics_report_on_test.csv", index=False)

    return metrics_df

