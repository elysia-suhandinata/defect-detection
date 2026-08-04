import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/severstal/train.csv')

all_images = sorted(os.listdir('data/severstal/train_images'))

labels = pd.DataFrame({'ImageId': all_images})
for c in [1, 2, 3, 4]:
    defect_images = set(df[df['ClassId'] == c]['ImageId'])
    labels[f'class_{c}'] = labels['ImageId'].isin(defect_images).astype(int)

labels['has_defect'] = labels[['class_1', 'class_2', 'class_3', 'class_4']].max(axis=1)

train_labels, val_labels = train_test_split(
    labels, test_size=0.2, random_state=42, stratify=labels['has_defect']
)

train_labels.to_csv('data/severstal/train_split.csv', index=False)
val_labels.to_csv('data/severstal/val_split.csv', index=False)

print('train:', len(train_labels), 'val:', len(val_labels))
print(train_labels[['class_1', 'class_2', 'class_3', 'class_4']].sum())