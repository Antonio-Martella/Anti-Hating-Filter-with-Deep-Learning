import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


def evaluation_class(count, folder = None):
  '''
  Generates and saves a bar chart with the class distribution.

  Parameters
  ----------
  count : pandas.Series
  Series containing the count of each class or category.
  folder : str, optional
  Subfolder of `results/` in which to save the chart (e.g., 'binary_hate').

  Output
  -------
  results/{folder}/distribution_class.png : image of the bar chart
  '''

  os.makedirs(f"results/{folder}", exist_ok=True)
  
  plt.figure(figsize=(6, 4))
  count.plot(kind='bar', color=sns.color_palette('viridis', len(count)))
  plt.title('Distribution of class')
  if folder == 'binary_hate':
    plt.xlabel('Hating (1) or not (0)')
  plt.ylabel('Count')
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.savefig(f"results/{folder}/distribution_class.png")
  plt.close()


# ---------------------------------------------------------------


def evaluate_model(model, X_test, y_test, threshold = 0.5, folder = None):
  '''
  Evaluates the model on the test set and saves the results in `results/{folder}`.

  Parameters
  ----------
  model : keras.Model
  Trained model to evaluate.
  X_test, y_test : array-like
  Test data and labels.
  threshold : float
  folder : str
  Subfolder to save reports and graphs (e.g., 'binary_hate').

  Output
  -------
  - metrics_report.csv : precision, recall, f1, support metrics
  - confusion_matrix.png : confusion matrix saved as an image
  ''' 
  y_pred = (model.predict(X_test) >= threshold).astype(int)

  report = classification_report(y_test, y_pred, output_dict=True)
  report_df = pd.DataFrame(report).transpose()
  os.makedirs(f"results/{folder}", exist_ok=True)
  report_df.to_csv(f'results/{folder}/metrics_report_on_test.csv', index=True)

  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(5, 4))
  if folder == 'binary_hate':
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Hate", "Hate"], yticklabels=["No Hate", "Hate"])
  elif folder == '':
    _
  plt.xlabel("Predict")
  plt.ylabel("Real")
  plt.title("Confusion Matrix")
  plt.savefig(f"results/{folder}/confusion_matrix.png")
  plt.close()
