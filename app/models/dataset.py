import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class SeverstalDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=256, synthetic_dir=None):
        self.labels = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.synthetic_dir = synthetic_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        if self.synthetic_dir is not None and 'synthetic' in row['ImageId']:
            img_path = os.path.join(self.synthetic_dir, row['ImageId'])
        else:
            img_path = os.path.join(self.img_dir, row['ImageId'])

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor([
            row['class_1'], row['class_2'], row['class_3'], row['class_4']
        ], dtype=torch.float32)

        return image, label