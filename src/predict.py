import numpy as np
import pandas as pd
import tensorflow as tf
import json
import pickle

from tensorflow.keras.models import load_model
from model import weighted_binary_crossentropy
from data_utils import  preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

from layers.attention import AttentionLayer


# -----------------------------
# -------- FIRST MODEL --------
# -----------------------------
# LOAD THE MODEL
print("\033[92m------ First Model ------\033[0m")
try:
  model_hate_binary = load_model(#'/models/binary_hate/model_hate_binary.h5',
    '/content/drive/MyDrive/Colab Notebooks/Progetto GitHub/DL GitHub/model_hate_binary.h5',
    custom_objects={"AttentionLayer": AttentionLayer})
  print(f"\033[92mFirst Model (model_hate_binary.h5) loaded successfully!\033[0m")
except Exception as e:
  print(f"\033[91mError loading First Model (model_hate_binary.h5): {e}!\033[0m")

# LOAD THE TOKENIZER 
try:
  with open("models/binary_hate/tokenizer_binary_hate.pkl", "rb") as f:
      tokenizer_binary_hate = pickle.load(f)
  print(f"\033[92mTokenizer for model_hate_binary loaded successfully!\033[0m")
except Exception as e:
  print(f"\033[91mError loading tokenizer of first model: {e}!\033[0m")

# LOAD THE OPTIMAL THRESHOLD FOR THE MODEL AND THE LENGHT FOR THE TOKENIZER
try:
    with open('models/binary_hate/param_model_binary_hate.json', 'r') as f:
        params = json.load(f)  
        max_len_bin_hate = params["max_len"]
        best_threshold_binary_hate = params["best_threshold"]
    print(f"\033[92mThe best threshold and tokenizer length loaded successfully!\033[0m")
except Exception as e:
    print(f"\033[91mError loading best threshold and tokenizer length of first model: {e}!\033[0m")

# ------------------------------
# -------- SECOND MODEL --------
# ------------------------------
# LOAD THE TENSOR WEIGHTS FOR THE 'model_hate_type'loaded_weights = np.load('results/hate_type/weights_tensor.npy')
'''weights_tensor = tf.constant(loaded_weights, dtype=tf.float32)

# Load the second model
try:
  model_hate_type = load_model(
    #'/models/hate_type/model_hate_type.h5',
    '/content/drive/MyDrive/Colab Notebooks/Progetto GitHub/DL GitHub/model_hate_type.h5',
    custom_objects={
      "AttentionLayer": AttentionLayer,
      "weighted_binary_crossentropy": weighted_binary_crossentropy(weights_tensor)
      },
      compile=False
    )
  print(f"\033[92mModel 'model_hate_type.h5' loaded successfully\033[0m")
except Exception as e:
  print(f"\033[91mError loading model 'model_hate_type.h5': {e}\033[0m")

# Load the optimal threshold for the first model
with open('results/binary_hate/best_threshold.json', 'r') as f:
  best_threshold_binary_hate = json.load(f)["threshold"]

# Load the optimal threshold for the first model
with open('models/binary_hate/tokenizer_param_binary_hate.json', 'r') as f:
  max_len_bin_hate = json.load(f)["max_len"]

# Load the tokenizer for the first model
try:
  with open("models/binary_hate/tokenizer_binary_hate.pkl", "rb") as f:
      tokenizer_binary_hate = pickle.load(f)
except Exception as e:
  print("Errore nel caricamento del tokenizer:", e)'''


df = pd.read_csv('data/binary_hate/test_binary_hate.csv')
df = preprocess_text(df, text_col="comment_text")

X = df[df["sum_injurious"] >= 1]
print(len(X))

# CORRETTO: passa solo la colonna dei testi
X_sequences = tokenizer_binary_hate.texts_to_sequences(X["comment_text"].astype(str))

padded_X_sequences = pad_sequences(
    sequences=X_sequences,
    maxlen=int(max_len_bin_hate)
)

y_pred = model_hate_binary.predict(padded_X_sequences)
y_pred_opt = (y_pred >= best_threshold_binary_hate).astype(int).flatten()

for i in range(50):
    print(X.sum_injurious.values[i], y_pred_opt[i])