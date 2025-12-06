import numpy as np
import pandas as pd
import tensorflow as tf
import json
import pickle

from tensorflow.keras.models import load_model
#from model import weighted_binary_crossentropy
from data_utils import  preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models import AttentionLayer
from utils import F1Score


# -----------------------------
# -------- FIRST MODEL --------
# -----------------------------
# LOAD THE MODEL
print("\033[92m------ First Model ------\033[0m")
try:
  model_binary_hate = load_model('models/binary_hate/model_hate_binary.keras')
  custom_objects={
        "AttentionLayer": AttentionLayer
        #"F1Score": F1Score
        }
  print(f"\033[92mFirst Model (model_binary_hate.keras) loaded successfully!\033[0m")
except Exception as e:
  print(f"\033[91mError loading First Model (model_binary_hate.keras): {e}!\033[0m")

# LOAD THE TOKENIZER 
try:
  with open("models/binary_hate/tokenizer_binary_hate.pkl", "rb") as f:
    tokenizer_binary_hate = pickle.load(f)
  print(f"\033[92mTokenizer for model_hate_binary loaded successfully!\033[0m")
except Exception as e:
  print(f"\033[91mError loading tokenizer of first model: {e}!\033[0m")

# LOAD TOKENAIZER PARAMETERS 
try:
    with open('models/binary_hate/param_tokenizer_binary_hate.json', 'r') as f:
        tokenizer_binary_hate_param = json.load(f)  
        max_len_binary_hate = tokenizer_binary_hate_param["max_len"]
        #vocabulary_size_binary_hate = tokenizer_binary_hate_param["vocabulary_size"]
    print(f"\033[92mTokenizer parameters loaded successfully!\033[0m")
except Exception as e:
    print(f"\033[91mError loading tokenizer parameters of first model: {e}!\033[0m")

# LOAD OPTIMIZED THRESHOLD
try:
    with open('models/binary_hate/best_thresholds_binary_hate.json', 'r') as f:
        best_threshold_binary_hate = json.load(f)  
    print(f"\033[92mOptimized threshold loaded successfully!\033[0m")
except Exception as e:
    print(f"\033[91mError loading optimized threshold of first model: {e}!\033[0m")


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


df = pd.read_csv('data/test_comments.csv')
df = preprocess_text(df, text_col="comment_text")

X = df[df["sum_injurious"] >= 1]
print(len(X))

X_sequences = tokenizer_binary_hate.texts_to_sequences(X["comment_text"].astype(str))

padded_X_sequences = pad_sequences(
    sequences=X_sequences,
    maxlen=int(max_len_binary_hate)
)

y_pred = model_binary_hate.predict(padded_X_sequences)
y_pred_opt = (y_pred >= best_threshold_binary_hate["best_thresholds_has_hate"]).astype(int).flatten()

for i in range(50):
    print(X.sum_injurious.values[i], y_pred_opt[i])