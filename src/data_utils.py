import pandas as pd
import tensorflow as tf
import math
import csv
import re
import os
from termcolor import colored
import pickle
import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(path: str = "../data/Filter_Toxic_Comments_dataset.csv") -> pd.DataFrame:
    """
    Loads the dataset and returns a pandas DataFrame.
    Checks that the file exists and is readable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            colored(f"The file {path} does not exist. "
            "Make sure you put it in the 'data/' folder.",'red')
        )
    
    try:
        df = pd.read_csv(path)
        print(colored(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns",'green'))
        return df
    except Exception as e:
        raise RuntimeError(colored(f"Error loading dataset: {e}",'red'))


#------------------------------------------------------------------


def preprocess_text(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """
    Best-practice text preprocessing for hate speech / sentiment models.

    Operations:
    - Lowercasing
    - Remove URLs
    - Remove mentions (@user)
    - Remove HTML tags
    - Remove non-alphanumeric EXCEPT punctuation useful for sentiment
    - Replace multiple spaces
    """
    def clean(text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        # Remove weird symbols but keep punctuation like ! ? . ,
        text = re.sub(r"[^a-zA-Z0-9.,!?\'\"\s]", " ", text)
        # Replace multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    df[text_col] = df[text_col].apply(clean)
    print(colored(f"Column '{text_col}' preprocessed successfully", "green"))
    return df


#------------------------------------------------------------------


def tokenization_and_pudding(x_train, x_test, num_words: int = None, verbose = False, folder = None):
    """
    Performs tokenization and padding on training and test texts.

    Args:
    x_train (list[str]): List of training texts.
    x_test (list[str]): List of test texts.
    num_words (int, optional): Maximum number of words to keep in the vocabulary. If None, all are considered.

    Returns:
    padded_train_sequences (np.ndarray): Training sequences with padding.
    padded_test_sequences (np.ndarray): Test sequences with padding.
    max_len (int): Maximum length of sequences.
    vocabulary_size (int): Vocabulary size.
    tokenizer (Tokenizer): Trained tokenizer object.
    """

    # Create and train the tokenizer
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x_train)

    # Converts texts to sequences of integers
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)

    # Determine the maximum length
    max_len = max(len(seq) for seq in train_sequences)

    # Directroy
    save_dir = f"../models/{folder}"

    # Create directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Save tokenizer parameters
    with open(os.path.join(save_dir, f"tokenizer_param_{folder}.json"), "w") as f:
        json.dump({"max_len": int(max_len)}, f, indent=4)

    # Apply padding
    padded_train_sequences = pad_sequences(sequences=train_sequences, maxlen=max_len)
    padded_test_sequences = pad_sequences(sequences=test_sequences, maxlen=max_len)

    # Calculate vocabulary size
    vocabulary_size = len(tokenizer.word_counts) + 1  # +1 for padding token

    # Save the tokenizer
    with open(os.path.join(save_dir, f"tokenizer_{folder}.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
      
    return padded_train_sequences, padded_test_sequences, max_len, vocabulary_size, tokenizer


#------------------------------------------------------------------


class CSVLoggerCustom(tf.keras.callbacks.Callback):
    """
    Custom callback to save training metrics to a CSV file.
    """
    def __init__(self, filename, verbose = False):
        super().__init__()
        self.filename = filename
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.file = open(self.filename, 'w', newline='')
        self.writer = None
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1'
        ])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def safe_f1(p, r):
            if p is None or r is None or (p + r) == 0 or math.isnan(p) or math.isnan(r):
                return None
            return 2 * p * r / (p + r)
        
        f1 = safe_f1(logs.get('precision'), logs.get('recall'))
        val_f1 = safe_f1(logs.get('val_precision'), logs.get('val_recall'))

        row = [
            epoch + 1,
            logs.get('loss'),
            logs.get('accuracy'),
            logs.get('precision'),
            logs.get('recall'),
            f1,
            logs.get('val_loss'),
            logs.get('val_accuracy'),
            logs.get('val_precision'),
            logs.get('val_recall'),
            val_f1,
        ]
        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, logs=None):
        self.file.close()
        if self.verbose == True:
            print(f"Training log saved in: {self.filename}")