import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from dataset import SeverstalDataset
from classifier import DefectCNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('using device:', device)

train_ds = SeverstalDataset(
    '../../data/severstal/train_split_vae_augmented.csv',
    '../../data/severstal/train_images',
    synthetic_dir='../../data/severstal/vae_synthetic'
)
val_ds = SeverstalDataset('../../data/severstal/val_split.csv', '../../data/severstal/train_images')

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = DefectCNN().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f'epoch {epoch+1}/{num_epochs} - train loss: {avg_train_loss:.4f} - val loss: {avg_val_loss:.4f}')

torch.save(model.state_dict(), 'vae_augmented_cnn.pth')
print('saved model to vae_augmented_cnn.pth')