import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_utils import load_dataset, preprocess_text, tokenization_and_pudding

def evaluate_model(model, X_test, y_test, fold = None):
    print("Predizioni in corso...")
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    print("Calcolo metriche...")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    os.makedirs(f"results/{fold}", exist_ok=True)
    report_df.to_csv(f'results/{fold}/metrics_report.csv', index=True)

    print("Generazione matrice di confusione...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Hate", "Hate"], yticklabels=["No Hate", "Hate"])
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Confusion Matrix")
    plt.savefig(f"results/{fold}/confusion_matrix.png")
    plt.close()

    print("Valutazione completata. Risultati salvati in 'results/'.")
