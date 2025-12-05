import re
import os
import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .param_utils import save_param


def preprocess_text(df: pd.DataFrame, text_col: str = "comment_text", verbose = False) -> pd.DataFrame:
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

    if verbose:
      print(f"\033[92mColumn '{text_col}' preprocessed successfully!\033[0m")

    return df


# ---------------------------------------------------


def tokenization_and_pad(X_train, X_test, num_words: int = None, verbose = False, folder = None):
    """
    Performs tokenization and padding on training and test texts.

    Args:
    X_train (list[str]): List of training texts.
    X_test (list[str]): List of test texts.
    num_words (int, optional): Maximum number of words to keep in the vocabulary. If None, all are considered.

    Returns:
    padded_train_sequences (np.ndarray): Training sequences with padding.
    padded_test_sequences (np.ndarray): Test sequences with padding.
    max_len (int): Maximum length of sequences.
    vocabulary_size (int): Vocabulary size.
    tokenizer (Tokenizer): Trained tokenizer object.
    """

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    # Converts texts to sequences of integers
    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    # Determine the maximum length
    max_len = max(len(seq) for seq in train_sequences)

    # Create directory if it does not exist
    os.makedirs(f"models/{folder}", exist_ok=True)

    # Save tokenizer parameters
    save_param(f"models/{folder}/param_tokenizer_{folder}.json", 
               "max_len", 
               int(max_len))

    # Apply padding
    padded_train_sequences = pad_sequences(sequences=train_sequences, maxlen=max_len)
    padded_test_sequences = pad_sequences(sequences=test_sequences, maxlen=max_len)

    # Effective vocabulary size (respects num_words)
    if num_words is None:
        vocabulary_size = len(tokenizer.word_index) + 1
        save_param(f"models/{folder}/param_tokenizer_{folder}.json", 
               "vocabulary_size", 
               int(vocabulary_size))
    else:
        vocabulary_size = min(num_words, len(tokenizer.word_index)) + 1
        save_param(f"models/{folder}/param_tokenizer_{folder}.json", 
                "vocabulary_size", 
                int(vocabulary_size))

    # Save the tokenizer
    with open(os.path.join(f"models/{folder}", f"tokenizer_{folder}.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
      
    return padded_train_sequences, padded_test_sequences, max_len, vocabulary_size, tokenizer



