import numpy as np
import pandas as pd
import tensorflow as tf
import json
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tensorflow.keras.models import load_model
<<<<<<< HEAD
from data_utils import  preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models import weighted_binary_crossentropy
=======
#from model import weighted_binary_crossentropy
from data_utils import  preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models import AttentionLayer
from utils import F1Score
>>>>>>> 6a61ec922e27f55239dbf64c4dbb510da2c9db32


# -----------------------------
# -------- FIRST MODEL --------
# -----------------------------
# LOAD THE MODEL
print("\033[92m------ First Model ------\033[0m")
try:
<<<<<<< HEAD
  model_binary_hate = load_model('models/binary_hate/model_hate_binary.h5')
=======
  model_binary_hate = load_model('models/binary_hate/model_hate_binary.keras')
  custom_objects={
        "AttentionLayer": AttentionLayer
        #"F1Score": F1Score
        }
>>>>>>> 6a61ec922e27f55239dbf64c4dbb510da2c9db32
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
loaded_weights = np.load('models/hate_type/weights_tensor.npy', allow_pickle=True)
weights_tensor = tf.constant(loaded_weights, dtype=tf.float32)

# Load the second model
try:
  model_hate_type = load_model(
    'models/hate_type/model_hate_type.h5',
    custom_objects={
      #"AttentionLayer": AttentionLayer,
      "weighted_binary_crossentropy": weighted_binary_crossentropy(weights_tensor)
    },
    compile=False
  )
  print(f"\033[92mModel 'model_hate_type.h5' loaded successfully\033[0m")
except Exception as e:
  print(f"\033[91mError loading model 'model_hate_type.h5': {e}\033[0m")

# Load the optimal threshold for the first model
with open('models/hate_type/best_thresholds_hate_type.json', 'r') as f:
  best_thresholds_hate_type = json.load(f)

# Load the optimal threshold for the first model
with open('models/hate_type/param_tokenizer_hate_type.json', 'r') as f:
  max_len_hate_type = json.load(f)["max_len"]

# Load the tokenizer for the first model
try:
  with open("models/hate_type/tokenizer_hate_type.pkl", "rb") as f:
      tokenizer_hate_type = pickle.load(f)
except Exception as e:
  print("Error loading second model tokenizer:", e)


<<<<<<< HEAD
while True:
    text = input("Inserisci una frase ('exit' per uscire): ")

    if text.lower() == "exit":
        break
    
    seq = tokenizer_binary_hate.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len_binary_hate)
    prob = model_binary_hate.predict(pad)[0][0]
    pred = int(prob >= best_threshold_binary_hate['best_thresholds_has_hate'])

    #print("Probabilità di hating:", prob)
    #print("Predizione:", pred)
    
    if pred == 1:
      seq = tokenizer_hate_type.texts_to_sequences([text])
      pad = pad_sequences(seq, maxlen=max_len_hate_type)
      prob = model_hate_type.predict(pad)[0]
      
      pred_labels = []
      labels_hate_type = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
      for i, label in enumerate(labels_hate_type):
          threshold = best_thresholds_hate_type[f"best_thresholds_{label}"]
          pred_labels.append(int(prob[i] >= threshold))
      print(pred_labels)
    else:
      print([0, 0, 0, 0, 0, 0])


## COSA RIMANE DA FARE:
## Creare un dataset come soli commenti, che viene passato
## al preprocess del testo (da implementare) e poi fornisce
## un file in unscita che identifica le labels, fare poi 
## un altro file di inferenza ma che accetta volta per volta
## dei commenti (vedere come implementare il preprocess del testo)


'''import numpy as np
import pandas as pd

# Carica dataset
df = pd.read_csv('data/train_and_test/test_dataset.csv')
=======
df = pd.read_csv('data/test_comments.csv')
>>>>>>> 6a61ec922e27f55239dbf64c4dbb510da2c9db32
df = preprocess_text(df, text_col="comment_text")

# Dividi tra hate / non-hate
df_hate = df[df["sum_injurious"] > 0]
df_nonhate = df[df["sum_injurious"] == 0]

<<<<<<< HEAD
# Seleziona 10 esempi casuali per ciascun gruppo
sample_hate = df_hate.sample(n=10, random_state=42)
sample_nonhate = df_nonhate.sample(n=10, random_state=42)
=======
X_sequences = tokenizer_binary_hate.texts_to_sequences(X["comment_text"].astype(str))
>>>>>>> 6a61ec922e27f55239dbf64c4dbb510da2c9db32

# Unisci in un unico df
samples = pd.concat([sample_hate, sample_nonhate]).reset_index(drop=True)

# --- PREPROCESSING PER IL MODELLO ---

X_sequences = tokenizer_binary_hate.texts_to_sequences(samples["comment_text"].astype(str))

padded_sequences = pad_sequences(
    sequences=X_sequences,
    maxlen=int(max_len_binary_hate)
)

<<<<<<< HEAD
# Predizioni
y_pred = model_binary_hate.predict(padded_sequences)
=======
y_pred = model_binary_hate.predict(padded_X_sequences)
>>>>>>> 6a61ec922e27f55239dbf64c4dbb510da2c9db32
y_pred_opt = (y_pred >= best_threshold_binary_hate["best_thresholds_has_hate"]).astype(int).flatten()

# --- STAMPA RISULTATI ---
for text, inj, pred in zip(samples["comment_text"], samples["sum_injurious"], y_pred_opt):
    print("---------------")
    print("TESTO:", text[:200], "..." if len(text) > 200 else "")
    print("sum_injurious:", inj)
    print("Predizione modello:", pred)'''
