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
├── data/                      # Dataset (raw e preprocessato). Non contiene file di grandi dimensioni.
├── notebooks/                 # Notebook di analisi esplorativa, esperimenti e prototipazione dei modelli.
├── src/
│   ├── preprocessing/         # Script per cleaning, tokenizzazione e preparazione testo
│   ├── models/                # Definizione delle architetture (binaria e multilabel)
│   ├── training/              # Script di training, valutazione e tuning dell'hyperparameters
│   ├── inference/             # Script per la predizione su nuovi commenti
│   └── utils/                 # Funzioni di supporto (metriche, callback, salvataggio/ caricamento modelli)
├── models/                    # Modelli salvati (pesI, tokenizer, best threshold, parametri)
├── results/                   # Metriche, curve di training, confusion matrix, report
├── requirements.txt           # Dipendenze per Windows/Linux/macOS Intel
├── requirements_macos_arm.txt # Dipendenze per macOS ARM (M1/M2/M3)
├── README.md                  # Documentazione del progetto
└── LICENSE                    # Licenza (opzionale)

```
Ogni componente del progetto è stato pensato per rendere il codice modulare e facilmente riutilizzabile:
- **`notebooks/`** contiene il flusso completo del progetto, utile per l’analisi e la presentazione.
- **`src/`** raccoglie il codice vero e proprio, strutturato in moduli indipendenti.
- **`data/`** permette di organizzare i dati di input senza appesantire il repository (i dataset completi non sono caricati su GitHub).
- **`models/`** include o descrive i modelli addestrati.
- **`results/`** mostra le prestazioni ottenute, utile per confrontare esperimenti diversi.


