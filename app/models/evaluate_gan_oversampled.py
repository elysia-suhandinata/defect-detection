import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import csv
from dataset import SeverstalDataset
from classifier import DefectCNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

val_ds = SeverstalDataset('../../data/severstal/val_split.csv', '../../data/severstal/train_images')
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = DefectCNN().to(device)
model.load_state_dict(torch.load('gan_oversampled_cnn.pth', map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
        all_preds.append(preds)
        all_labels.append(labels.int())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

target_names = ['class_1', 'class_2', 'class_3', 'class_4']

report_dict = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

results_path = '../../results/results.csv'
with open(results_path, 'a', newline='') as f:
    writer = csv.writer(f)
    for cls in target_names + ['macro avg']:
        row = report_dict[cls]
        writer.writerow(['gan_oversampled', cls.replace(' ', '_'), round(row['precision'], 4), round(row['recall'], 4), round(row['f1-score'], 4)])

print('gan_oversampled results appended to results/results.csv')