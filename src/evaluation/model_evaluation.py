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



def evaluate_model(model, X_padded, y_true, labels, folder = None):

  # best threshold
  y_pred = model.predict(X_padded)
  optimal_thresholds = f1_score_optimization_thresholds(y_true, y_pred, labels, folder)

  # result
  n_classes = len(labels)
  metrics_data = [] 

  for i in range(n_classes):
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

  path = f"results/{folder}"
  os.makedirs(path, exist_ok=True)
  metrics_df.to_csv(f"{path}/metrics_report_on_test.csv", index=True)
  
