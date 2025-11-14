# Dataset

This project uses a dataset of 159571 generic text comments labeled across six different hate types.

## Content
In this folder there is the file:
- **`Filter_Toxic_Comments_dataset.csv`** — contains text comments and their labels (`0` = not offensive, `1` = offensive) across six hate categories.

## Dataset description
Each line in the file represents an example, with the following main columns:

| Column | Description |
|-----------------|------------|
| `comment_text`| Textual content of the comment or phrase |
| `toxic`| Binary tag for toxic content |
| `severe_toxic`| Binary tag for highly toxic content |
| `obscene`| Binary tag for obscene language |
| `threat`| Binary tag for threats |
| `insult`| Binary tag for insults |
| `identity_hate`| Binary tag for hatred towards specific identities |
| `sum_injurious`| Sum of previous tags |


## Source
The dataset is available on Google Drive at the following link:

[Download the dataset from Google Drive](https://drive.google.com/file/d/1Y0750AWh4Wp6M1WOOxaNH0Rv7hy9BlUx/view?usp=drive_link)

*(The file is shared read-only; anyone can download it.)*

## Usage
The main notebook (`notebooks/Anti-Hating_Filter.ipynb`) automatically reads the file:
```python
import pandas as pd
df = pd.read_csv("data/Filter_Toxic_Comments_dataset.csv")
```

