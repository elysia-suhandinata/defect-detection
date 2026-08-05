import pandas as pd

train = pd.read_csv('data/severstal/train_split.csv')

mask = (train['class_1'] == 1) | (train['class_2'] == 1) | (train['class_4'] == 1)
vae_labels = train[mask]

vae_labels.to_csv('data/severstal/vae_train_labels.csv', index=False)
print('vae training images:', len(vae_labels))
print(vae_labels[['class_1', 'class_2', 'class_4']].sum())