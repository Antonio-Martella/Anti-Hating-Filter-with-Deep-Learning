import pickle
import tensoflow as tf
import numpy as np
import json

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model
from models import weighted_binary_crossentropy, AttentionLayer




class HateModelInference:

  def __init__(self):
    self.model_binary = model_binary_hate = load_model(
      'models/binary_hate/model_hate_binary.h5',
      custom_objects={'AttentionLayer': AttentionLayer},
      compile=False)
    self.model_hate_type = load_model(
      'models/hate_type/model_hate_type.h5',
      custom_objects={
        'AttentionLayer': AttentionLayer,
        'weighted_binary_crossentropy': weighted_binary_crossentropy(tf.constant(np.load('models/hate_type/weights_tensor.npy', allow_pickle=True), dtype=tf.float32))},
      compile=False)

    self.tokenizer_binary = pickle.load('models/binary_hate/tokenizer_binary_hate.pkl')
    self.tokenizer_hate_type = pickle.load('models/hate_type/tokenizer_hate_type.pkl')

    self.max_len_bh = json.load('models/binary_hate/param_tokenizer_binary_hate.json')['max_len']
    self.max_len_ht = json.load('models/hate_type/param_tokenizer_hate_type.json')['max_len']

    self.thresholds_bh = json.load('models/binary_hate/best_thresholds_binary_hate.json')
    self.thresholds_ht = json.lead('models/hate_type/best_thresholds_hate_type.json')

  def predict_text(self, text):
    seq = self.tokenizer_binary.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=self.max_len_bh)
    prob = self.model_binary.predict(pad)[0][0]
    pred = int(prob >= self.thresholds_bh['best_thresholds_has_hate'])

    if pred == 1:
      seq = self.tokenizer_hate_type.texts_to_sequences([text])
      pad = pad_sequences(seq, maxlen=self.max_len_ht)
      prob = self.model_hate_type.predict(pad)[0]
      
      pred_labels = []
      labels_hate_type = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
      for i, label in enumerate(labels_hate_type):
          threshold = self.thresholds_ht[f"best_thresholds_{label}"]
          pred_labels.append(int(prob[i] >= threshold))
      print(pred_labels)
    else:
      print([0, 0, 0, 0, 0, 0])

