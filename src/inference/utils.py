from tensorflow.keras.models import load_model
from models import weighted_binary_crossentropy, AttentionLayer


class HateModelInference:
    def __init__(self):
        self.model_binary = model_binary_hate = load_model(
          'models/binary_hate/model_hate_binary.h5',
          custom_objects={"AttentionLayer": AttentionLayer},
    compile=False
  )
        self.model_hate_type = load_model(...)
        self.tokenizer_binary = ...
        self.tokenizer_hate_type = ...
        self.thresholds = ...

    def predict_text(self, text):
        # return dict con: prob, binary_label, multilabels
