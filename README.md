# Deep Learning Project – Toxic Comment Classification

## Overview
Questo progetto nasce con l’obiettivo di sviluppare un sistema di classificazione automatica dei commenti tossici utilizzando tecniche di Deep Learning applicate al Natural Language Processing (NLP).

L’intero workflow – dalla preparazione del dataset alla progettazione dei modelli, dall’ottimizzazione degli iperparametri alla valutazione finale – è stato progettato e implementato da me, con l’intento di mostrare competenze solide in:
 
- Machine Learning e Deep Learning

- NLP e text preprocessing

- Architetture basate su reti ricorrenti (LSTM)

- Ottimizzazione, validazione e gestione dei modelli

- Buone pratiche di MLOps a livello progettuale (struttura repository, riproducibilità, separazione dei moduli)

## Obiettivo del progetto

L’obiettivo è costruire un modello robusto in grado di:

1. Identificare se un commento contiene contenuti d’odio (classificazione binaria).

2. Classificare le specifiche categorie di tossicità presenti nel commento (classificazione multilabel), tra cui:

	- toxic

	- severe_toxic

	- insult

	- threat

	- identity_hate

	- obscene

## Architettura del sistema

Il sistema è organizzato in due stadi:

1. Modello Binario

	- Distingue commenti neutri da commenti che presentano qualunque tipo di tossicità.

	- Ottimizzato per ridurre i falsi negativi (non perdere commenti tossici).


2. Modello Multilabel

	- Viene attivato solo quando il primo modello segnala la presenza di tossicità.

	- È addestrato esclusivamente sui commenti tossici, per distinguere le diverse categorie.

Questa struttura “a cascata” migliora performance, efficienza e interpretabilità dei risultati.


## Struttura del progetto

La struttura del repository è organizzata per separare in modo chiaro i diversi componenti del workflow di Deep Learning:
```
.
├── data/                      # Dataset per training e inferenza
├── notebooks/                 # Notebook di analisi esplorativa, esperimenti e prototipazione dei modelli.
├── src/
│   ├── preprocessing/         # Script per cleaning, tokenizzazione e preparazione testo
│   ├── models/                # Definizione delle architetture (binaria e multilabel)
│   ├── training/              # Script di training e tuning dell'hyperparameters
│   ├── inference/             # Script per la predizione su nuovi commenti
│   └── utils/                 # Funzioni di supporto (metriche, callback, salvataggio/ caricamento modelli)
├── models/                    # Modelli salvati (pesi, tokenizer, best threshold, parametri)
├── results/                   # Metriche, curve di training, confusion matrix, report
├── requirements.txt           # Dipendenze per Windows/Linux/macOS Intel
├── requirements_macos_arm.txt # Dipendenze per macOS ARM (M1/M2/M3)
├── README.md                  # Documentazione del progetto
└── LICENSE                    # Licenza

```

## Installation
Clonare la repository:
```bash
git clone https://github.com/Antonio-Martella/Anti-Hating-Filter-with-Deep-Learning.git
cd Anti-Hating-Filter-with-Deep-Learning/
```
Creare e attivare l'ambiente virtuale:
```bash
# Linux/MacOS
python -m venv venv
source venv/bin/activate
```
```bash
# Windows
python -m venv venv
venv\Scripts\activate    
```
Installare le dipendenze:
```bash
# Windows/Linux/macOS (Intel/ARM)
pip install -r requirements.txt
```
```bash
# MacOS ARM (Apple Silicon)
pip install -r requirements_macos_arm.txt
```

## Usage

###  Training e Tuning da zero
Se vuoi ottimizzare gli iperparametri e allenare i modelli da zero:

**Tuning degli iperparametri**
```bash
# Modello binary_hate
python src/training/binary_hate/optuna_search_bh.py

# Modello hate_type
python src/training/hate_type/optuna_search_ht.py
```

**Training dei modelli con i best hyperparameters**
```bash
# Modello binary_hate
python src/training/binary_hate/train_binary_hate_model.py

# Modello hate_type
python src/training/hate_type/train_hate_type_model.py
```

### 2️⃣Usare modelli pre-addestrati

Se vuoi risparmiare tempo e non allenare i modelli da zero, puoi scaricare i modelli già addestrati e usarli direttamente per l’inferenza.

**1. Scarica i modelli dai link del Drive:**
- [model_binary_hate.keras](https://drive.google.com/file/d/1o3QSmFB2EIus8QugOYzuapOQB81vP2Gl/view?usp=drive_link) → posizionalo in `models/binary_hate/`
- `hate_type.keras` → posizionalo in `models/hate_type/`

**2. Esegui l’inferenza su un commento di esempio**
```bash
python src/inference/predict.py --text "You are an idiot"

