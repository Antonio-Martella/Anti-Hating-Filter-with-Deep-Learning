# Progetto Deep Learning – Classificazione Commenti Tossic

Questo progetto mostra un flusso completo di *Deep Learning* per classificare commenti testuali (binary e multilabel).  
Include: preprocessing, addestramento, validazione e inferenza finale.


## Struttura del progetto

La struttura del repository è organizzata per separare in modo chiaro i diversi componenti del workflow di Deep Learning:

'''
Progetto_DL/
├── notebooks/ # Contiene i notebook Jupyter (.ipynb) con EDA, training e test
│ ├── Progetto_DL.ipynb # Notebook principale: analisi, preprocessing, training, validazione e inferenza
│ └── demo_inference.ipynb # (Facoltativo) notebook demo con poche predizioni pronte da mostrare
│
├── src/ # Moduli Python che contengono il codice eseguibile
│ ├── data_utils.py # Funzioni per il caricamento e preprocessing del dataset
│ ├── model.py # Definizione dell’architettura del modello Keras/TensorFlow
│ ├── train.py # Script per addestrare il modello e salvare i pesi
│ ├── evaluate.py # Funzioni di valutazione e metriche
│ └── predict.py # Script per lanciare inferenza su nuovi dati
│
├── data/ # Cartella dedicata ai dati
│ ├── README.md # Spiega come ottenere o scaricare il dataset
│ └── sample.csv # (Facoltativo) piccolo esempio di dati di input
│
├── models/ # Modelli addestrati o link per scaricarli
│ └── README.md # Spiega come scaricare i pesi del modello
│
├── results/ # Risultati sperimentali (grafici, tabelle, log)
│ ├── training_curves.png # Esempio: curva loss/accuracy
│ ├── confusion_matrix.png # Esempio: matrice di confusione
│ └── metrics_report.csv # Risultati numerici
│
├── requirements.txt # Librerie Python necessarie al progetto
├── LICENSE # Licenza MIT
├── .gitignore # File che indica a Git cosa non deve essere incluso
└── README.md # File principale di documentazione del progetto
'''

Ogni componente del progetto è stato pensato per rendere il codice modulare e facilmente riutilizzabile:
- **`notebooks/`** contiene il flusso completo del progetto, utile per l’analisi e la presentazione.
- **`src/`** raccoglie il codice vero e proprio, strutturato in moduli indipendenti.
- **`data/`** permette di organizzare i dati di input senza appesantire il repository (i dataset completi non sono caricati su GitHub).
- **`models/`** include o descrive i modelli addestrati.
- **`results/`** mostra le prestazioni ottenute, utile per confrontare esperimenti diversi.


