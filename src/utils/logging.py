import os
import csv
import math
import tensorflow as tf

class CSVLoggerCustom(tf.keras.callbacks.Callback):
    def __init__(self, filename, metrics=None, verbose=False):
        super().__init__()
        self.filename = filename
        self.verbose = verbose
        self.metrics = metrics or [
            'loss', 'accuracy', 'precision', 'recall', 'f1',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1'
        ]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)

    def on_train_begin(self, logs=None):
        self.writer.writerow(['epoch'] + self.metrics)

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
            val_f1
        ]
        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, logs=None):
        self.file.close()
        if self.verbose:
            print(f"Training log saved in: {self.filename}")
