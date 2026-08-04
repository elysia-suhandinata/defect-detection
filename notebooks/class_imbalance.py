import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('data/severstal/train.csv')

total_images = len(os.listdir('data/severstal/train_images'))
images_with_defect = df['ImageId'].nunique()
images_no_defect = total_images - images_with_defect

print('total images:', total_images)
print('images with at least one defect:', images_with_defect)
print('images with no defect:', images_no_defect)
print()

class_counts = df['ClassId'].value_counts().sort_index()
print('defect instances per class:')
print(class_counts)
print()

defects_per_image = df.groupby('ImageId')['ClassId'].nunique()
multi_defect_images = (defects_per_image > 1).sum()
print('images with more than one defect class:', multi_defect_images)

labels = ['no defect'] + [f'class {c}' for c in class_counts.index]
counts = [images_no_defect] + class_counts.tolist()

plt.figure(figsize=(8, 5))
plt.bar(labels, counts)
plt.ylabel('number of images')
plt.title('severstal class distribution')
plt.tight_layout()
plt.savefig('notebooks/class_distribution.png')
print()
print('saved chart to notebooks/class_distribution.png')