import os
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluate import evaluation_class


def split_dataset_binary(df, test_size=0.2, val_size = 0.2, stratify=True, augmentation=False):

  df = df.copy()
  df['has_hate'] = (df['sum_injurious'] > 0).astype(int)

  class_counts = df['has_hate'].value_counts().sort_index()
  evaluation_class(count=class_counts, folder='binary_hate')

  train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=1,
    stratify=df['has_hate'] if stratify else None,
    shuffle=True
  )

  train_df, val_df = train_test_split(
    train_df,
    test_size=test_size,
    random_state=1,
    stratify=train_df['has_hate'] if stratify else None,
    shuffle=True
  )

  path = '../data/binary_hate'
  os.makedirs(path, exist_ok=True)
  (train_df.drop(columns='has_hate', errors="ignore").to_csv(f"{path}/train_binary_hate.csv", index=False))
  (test_df.drop(columns='has_hate', errors="ignore").to_csv(f"{path}/test_binary_hate.csv", index=False))
  (val_df.drop(columns='has_hate', errors="ignore").to_csv(f"{path}/val_binary_hate.csv", index=False))

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

  return train_aug, test_df, val_df


#------------------------------------------------------------------


def split_dataset_hate_type(df, test_size=0.2, val_size = 0.2):
  
  df = df[df["sum_injurious"] >= 1]
  print(len(df))

  class_counts = df.loc[:, 'toxic':'identity_hate'].value_counts().sort_index()
  evaluation_class(count=class_counts, folder='hate_type')

  train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=1,
    shuffle=True
  )
    
  train_df, val_df = train_test_split(
    train_df,
    test_size=test_size,
    random_state=1,
    shuffle=True
  )
    
  path = '../data/hate_type'
  os.makedirs(path, exist_ok=True)
  train_df.to_csv(f"{path}/train_hate_type.csv", index=False)
  test_df.to_csv(f"{path}/test_hate_type.csv", index=False)
  val_df.to_csv(f"{path}/val_hate_type.csv", index=False)

  return train_df, test_df, val_df
