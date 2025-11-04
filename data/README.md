# Dataset

Questo progetto utilizza un dataset di testo per l’addestramento del filtro anti-hating.

## Contenuto
In questa cartella è presente il file:
- `Filter_Toxic_Comments_dataset.csv` — contiene i dati testuali e le relative etichette (`0` = non offensivo, `1` = offensivo) alle sei differenti labels che caratterizzano la tipologia di odio (se presente) del testo.

## Descrizione del dataset
Ogni riga del file rappresenta un esempio con le seguenti colonne principali:
## 🧾 Descrizione del dataset
Ogni riga del file rappresenta un esempio, con le seguenti colonne principali:
```
| Colonna | Descrizione |
|----------|-------------|
| `comment_text` | Contenuto testuale del commento o frase |
| `toxic` | Etichetta binaria per contenuti tossici |
| `severe_toxic` | Etichetta per contenuti altamente tossici |
| `obscene` | Etichetta per linguaggio osceno |
| `threat` | Etichetta per minacce |
| `insult` | Etichetta per insulti |
| `identity_hate` | Etichetta per odio verso identità specifiche |
| `sum_injurious` | Somma p delle etichette precedenti |
```

## Origine
Il dataset è scaricabile dal mio drive personale
[link](https://drive.google.com/file/d/1Y0750AWh4Wp6M1WOOxaNH0Rv7hy9BlUx/view?usp=drive_link) 

## Utilizzo
Il notebook principale (`notebooks/Anti-Hating_Filter.ipynb`) legge automaticamente il file:
```python
import pandas as pd
data = pd.read_csv("data/Filter_Toxic_Comments_dataset.csv")
```

