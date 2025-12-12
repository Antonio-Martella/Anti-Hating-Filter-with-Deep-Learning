import os, sys, pickle, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from src.models import weighted_binary_crossentropy
from src.inference.utils import clean


class HateModelInference:
  
  '''
  Initializes the full inference stack for the hate-detection system.

  This constructor loads all assets required at prediction time, including:
  - The binary hate model and the multi-label hate-type model.
  - Their tokenizers and preprocessing configurations.
  - Class-specific optimal thresholds for post-processing.
  - All associated metadata such as maximum sequence lengths.

  Inputs
      None. All resource paths are automatically derived from the project root.

  Outputs
      A fully initialized HateModelInference object equipped with:
        – Loaded TensorFlow models for both tasks.
        – Tokenizers for text-to-sequence transformations.
        – Maximum input lengths used during training.
        – Threshold dictionaries for converting probabilities into class labels.
  '''

  def __init__(self):

    # ROOT PROJECT
    self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # MODEL PATHS
    self.path_binary = os.path.join(self.root, "models/binary_hate")
    self.path_hate_type = os.path.join(self.root, "models/hate_type")

    # LOAD BINARY MODEL
    self.model_binary = load_model(
      os.path.join(self.path_binary, "model_hate_binary.h5"),
      compile=False
    )

    #LOAD HATE TYPE
    weights = np.load(os.path.join(self.path_hate_type, "weights_tensor.npy"), allow_pickle=True)
    loss_fn = weighted_binary_crossentropy(tf.constant(weights, dtype=tf.float32))

    self.model_hate_type = load_model(
      os.path.join(self.path_hate_type, "model_hate_type.h5"),
      custom_objects={
        "weighted_binary_crossentropy": loss_fn
      },
      compile=False
    )

    # LOAD THE TOKENIZERS
    self.tokenizer_binary = pickle.load(open(
      os.path.join(self.path_binary, "tokenizer_binary_hate.pkl"), "rb"
    ))

    self.tokenizer_hate_type = pickle.load(open(
      os.path.join(self.path_hate_type, "tokenizer_hate_type.pkl"), "rb"
    ))

    # LOAD MAX LENGHTS 
    self.max_len_bh = json.load(open(
      os.path.join(self.path_binary, "param_tokenizer_binary_hate.json")
    ))["max_len"]

    self.max_len_ht = json.load(open(
      os.path.join(self.path_hate_type, "param_tokenizer_hate_type.json")
    ))["max_len"]

    # LAOD THRESHOLDS
    self.thresholds_bh = json.load(open(
      os.path.join(self.path_binary, "best_thresholds_binary_hate.json")
    ))

    self.thresholds_ht = json.load(open(
      os.path.join(self.path_hate_type, "best_thresholds_hate_type.json")
    ))



  def predict_text(self, text, verbose=True):

    # DATA CLEANING
    text = clean(text)

    # FIRST MODEL (binary hate)
    seq = self.tokenizer_binary.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=self.max_len_bh)
    prob = self.model_binary.predict(pad, verbose = 0)[0][0]

    pred = int(prob >= self.thresholds_bh["best_thresholds_has_hate"])

    if pred == 0:
      return [0,0,0,0,0,0]

    # SECOND MODEL (hate type)
    seq = self.tokenizer_hate_type.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=self.max_len_ht)
    probs = self.model_hate_type.predict(pad, verbose = 0)[0]

    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    outputs = [
      int(probs[i] >= self.thresholds_ht[f"best_thresholds_{label}"])
      for i, label in enumerate(labels)
    ]

    return outputs
