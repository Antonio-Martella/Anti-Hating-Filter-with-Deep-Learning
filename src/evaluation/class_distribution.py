import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_distribution(count, folder=None):
  
    '''
    Generates and saves a bar chart showing the distribution of classes in a dataset.

    Parameters
    ----------
    count : pandas.Series
        Series containing the count of each class or category.
    folder : str, optional
        Subfolder of "results/" where the chart will be saved (e.g., 'binary_hate').

    Output
    ------
    Saves a PNG file in "results/{folder}/distribution_class.png" representing class frequencies.
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