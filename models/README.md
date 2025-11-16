# Available Models

This project includes two trained neural network models:

## 1. Hate Comment Binary Classifier

This model determines whether a given comment should be classified as hate speech or non-hate.
You can download the pretrained weights here:
[Google Drive –` model_hate_binary.h5`](https://drive.google.com/file/d/1SdJGPULlTyfMU8klkSIqE-J78murGaWq/view?usp=drive_link)

Place the downloaded file inside the /models directory (this folder).

## 2. Hate Type Multi-Label Classifier
If a comment is classified as hateful by the first model, this second model identifies which of the six hate categories it belongs to.

You can download the pretrained weights here:

[Google Drive – `model_hate_type.h5`](https://drive.google.com/file/d/1iPeJC9k5SVW7ifqz2De5X2EEgAvEcOkI/view?usp=drive_link)
After downloading, place the file in the /models directory.

Alternatively, you can generate the models from scratch by running:
```
python src/main.py
```
