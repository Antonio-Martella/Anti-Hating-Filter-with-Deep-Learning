import os
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluation import plot_class_distribution


def split_dataset_binary(df, test_size=0.2, stratify=True, augmentation=False):

  df = df.copy()
  df['has_hate'] = (df['sum_injurious'] > 0).astype(int)

  class_counts = df['has_hate'].sum().sort_values(ascending=False)
  plot_class_distribution(count=class_counts, folder='binary_hate')

  train_df, test_df = train_test_split(
    df,
    test_size = test_size,
    random_state = 1,
    stratify = df['has_hate'] if stratify else None,
    shuffle = True
  )


  path = 'data/binary_hate'
  os.makedirs(path, exist_ok=True)
  (train_df.drop(columns='has_hate', errors="ignore").to_csv(f"{path}/train_binary_hate.csv", index=False))
  (test_df.drop(columns='has_hate', errors="ignore").to_csv(f"{path}/test_binary_hate.csv", index=False))

  if augmentation:
    augmented_rows = []
    for _, row in train_df.iterrows():
      s = int(row['sum_injurious'])
      if s <= 1:
        repeat_n = 1
      else:
        repeat_n = s 
      for _ in range(repeat_n):
        augmented_rows.append(row.copy())
    train_aug = pd.DataFrame(augmented_rows)
    train_aug = train_aug.sample(frac=1, random_state=1).reset_index(drop=True)
  else:
      train_aug = train_df.copy()

  return train_aug, test_df


#------------------------------------------------------------------


def split_dataset_hate_type(df, test_size=0.2):

  path = 'data/train_and_test'

  try:
    df_train = pd.read_csv(f'{path}/train_dataset.csv')
    df_test = pd.read_csv(f'{path}/test_dataset.csv')
  except Exception as e:
    raise RuntimeError(f"\033[91mError loading dataset: {e}.\033[0m")
  
  df_train = df_train[df_train["sum_injurious"] >= 1]
  df_test = df_test[df_test["sum_injurious"] >= 1]

  class_counts = df_train.loc[:, 'toxic':'identity_hate'].sum().sort_values(ascending=False)
  plot_class_distribution(count=class_counts, folder='hate_type')

  return df_train, df_test
