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

from .f1_threshold_optimization import f1_score_optimization
from .class_distribution import plot_class_distribution


def evaluate_model(model, X_test, y_test, folder=None):

    '''
    Evaluates a trained model on a test set using an optimized threshold.
    Computes performance metrics, saves a classification report and a confusion matrix (for binary tasks).

    Parameters
    ----------
    model : keras.Model
        The trained Keras model to evaluate.
    X_test : array-like
        Input features for the test set (e.g., padded sequences for NLP models).
    y_test : array-like
        True labels corresponding to X_test (binary or multi-class).
    folder : str, optional
        Subfolder of "results/" where metrics and figures will be saved (e.g., 'binary_hate').

    Output
    ------
    - CSV file with precision, recall, F1-score, and support: "results/{folder}/metrics_report_on_test.csv"
    - PNG confusion matrix for binary classification: "results/{folder}/confusion_matrix.png"
    '''

    count = df['has_hate'].value_counts().sort_index()
    plot_class_distribution(count, folder='binary_hate')

    y_pred = model.predict(X_test)

    optimal_threshold = f1_score_optimization(y_test, y_pred, folder)
    y_pred_opt = (y_pred >= optimal_threshold).astype(int).flatten()

    report = classification_report(y_test, y_pred_opt, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    os.makedirs(f"results/{folder}", exist_ok=True)
    report_df.to_csv(f"results/{folder}/metrics_report_on_test.csv", index=True)

    # Confusion matrix for binary model
    if folder == 'binary_hate':
        cm = confusion_matrix(y_test, y_pred_opt)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["No Hate", "Hate"],
                    yticklabels=["No Hate", "Hate"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(f"results/{folder}/confusion_matrix.png")
        plt.close()