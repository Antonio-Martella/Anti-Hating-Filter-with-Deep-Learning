# Deep Learning Project – Toxic Comment Classification

## Overview
This project was born with the aim of developing an automatic classification system for toxic comments using Deep Learning techniques applied to Natural Language Processing (NLP).

The entire workflow – from dataset preparation to model design, from hyperparameter optimization to final evaluation – was designed and implemented by me, with the aim of demonstrating solid skills in:
 
- Machine Learning and Deep Learning

- NLP and Text Preprocessing

- Recurrent Network (LSTM) Architectures

- Model Optimization, Validation, and Management

- MLOps Best Practices at the Design Level (Repository Structure, Reproducibility, Module Separation)

## Project objective

The goal is to build a robust model capable of:

1. Identifying whether a comment contains hate content (binary classification).

2. Classifying the specific toxicity categories present in the comment (multilabel classification), including:

	- toxic

	- severe_toxic

	- insult

	- threat

	- identity_hate

	- obscene

## System Architecture

The system is organized into two stages:

1. Binary Model

	- Distinguishes neutral comments from comments with any type of toxicity.

	- Optimized to reduce false negatives (don't miss toxic comments).

2. Multilabel Model

	- It is activated only when the first model reports the presence of toxicity.

	- It is trained exclusively on toxic comments, to distinguish between different categories.

This "cascade" structure improves performance, efficiency, and interpretability of results.

## Neural Network Architecture and Design Rationale
The design of both models was driven by the characteristics of the task, the linguistic nature of short informal comments, and the need to maximize robustness while keeping computational efficiency.

### Binary Classification Model

**Goal**: detect whether a comment contains any toxic content.

**Architecture**:

- Embedding layer 

- Bidirectional LSTM

- Dense

- Dense classifiers

**Reasons for this design**:

1. **Sequential dependencies matter**: Toxic expressions often depend on the interplay of words and their local context. The Bidirectional LSTM captures long-range dependencies in both directions.

2. **Efficiency and robustness**: A single-layer BiLSTM with attention provides an optimal balance between expressive power and low inference cost, suitable for real-time toxicity detection.


### Multilabel Toxicity Classification Model
**Goal**: identify which specific categories of toxicity are present in a comment.

**Architecture**:

- Embedding layer

- Bidirectional LSTM

- Dense

- Multi-head dense output (6 sigmoid units, one per class)

NB: Custom weighted loss (to compensate class imbalance)

**Reasons for this design**:

1. **Multilabel nature of the task**: A comment may contain multiple forms of toxicity simultaneously (e.g., insult + obscene + threat). Using sigmoid outputs instead of softmax allows independent probabilities per class.

2. **Class imbalance handling**: The dataset contains rare classes (e.g., threat or severe_toxic). A custom weighted binary cross-entropy ensures that rare classes contribute more to the gradient, improving recall.

3. **Hierarchical modeling**: The second model is trained only on toxic comments, reducing noise and helping the network specialize on the nuances of toxic subcategories.


## Project Structure

The repository structure is organized to clearly separate the different components of the Deep Learning workflow:

```
.
├── data/                      # Dataset for training and inference
├── notebooks/                 # Exploratory analysis notebook, experiments and model prototyping.
├── src/
│   ├── evaluation/            # Script for model evaluation
│   ├── models/                # Architecture definition (binary and multilabel)
│   ├── training/              # Hyperparameter tuning and model training script
│   ├── inference/             # # Script for predicting new comments (single sentences or .csv files)
│   └── utils/                 # Support functions (metrics, tokenization, cleaning, splitting, etc.)
├── models/                    # Trained model files, .json of best hyperparameters, best thresholds, tokenizer
├── results/                   # Metrics, confusion matrix, reports, output .csv files
├── requirements.txt           # Dependencies for Windows/Linux/macOS Intel
├── requirements_macos_arm.txt # Dependencies for macOS ARM (M1/M2/M3)
├── README.md                  # Detailed description of the repository structure
└── LICENSE                    # License

```

## Installation
Clone the repository:
```bash
git clone https://github.com/Antonio-Martella/Anti-Hating-Filter-with-Deep-Learning.git
cd Anti-Hating-Filter-with-Deep-Learning/
```
Creating and activating the virtual environment, for Linux/MacOS
```bash
python -m venv venv
source venv/bin/activate
```
for Windows
```bash
python -m venv venv
venv\Scripts\activate    
```
Install dependencies, for Windows/Linux/MacOS (Intel)
```bash
pip install -r requirements.txt
```
for MacOs ARM (Apple Silicon)
```bash
pip install -r requirements_macos_arm.txt
```

## Usage

### Training and Tuning from Scratch
Training the models from scratch, and especially performing hyperparameter optimization, requires significant computational resources. For this reason, we recommend using a dedicated GPU or cloud-based solutions.

The simplest option is Google Colab, which is the environment used during development. With an L4 GPU, the full hyperparameter search and model training complete in approximately one hour (using the search ranges provided in this repository).

If you want to reproduce the full workflow from scratch, including hyperparameter optimization and training (all seeds are fixed to ensure reproducibility), you can simply run:

**Tuning degli iperparametri**  
Binary hate model
```bash
python src/training/binary_hate/optuna_search_bh.py
```
Hate type model
```bash
python src/training/hate_type/optuna_search_ht.py
```

**Training models with the best hyperparameters**  
Binary hate model
```bash
python src/training/binary_hate/train_binary_hate_model.py
```
Hate type model
```bash
python src/training/hate_type/train_hate_type_model.py
```

### Using pre-trained models

If you want to save time and avoid training models from scratch, you can download pre-trained models and use them directly for inference.

**Download the templates from the Drive links:**
- [model_binary_hate.h5](https://drive.google.com/file/d/10c-YEeP1nxWQeWUiqdfj0MH9DumaghUV/view?usp=drive_link) → place it in `models/binary_hate/`
- [model_hate_type.h5](https://drive.google.com/file/d/1osNGAY8DDs2SiC-a4sH0GtD6e0DBF52M/view?usp=drive_link) → place it in `models/hate_type/`

### Inferenza 
Per eseguire l'inferenza su un commento di esempio:
```bash
python src/inference/predict.py --text "You are an idiot"


