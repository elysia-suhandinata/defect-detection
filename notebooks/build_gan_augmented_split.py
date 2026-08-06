import pandas as pd

real = pd.read_csv('data/severstal/train_split.csv')
synthetic = pd.read_csv('data/severstal/gan_synthetic_labels.csv')

combined = pd.concat([real, synthetic], ignore_index=True)
combined.to_csv('data/severstal/train_split_gan_augmented.csv', index=False)

print('real:', len(real), 'synthetic:', len(synthetic), 'combined:', len(combined))