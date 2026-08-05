import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from dataset_vae import VAEDataset
from cvae import ConditionalVAE

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('using device:', device)

train_ds = VAEDataset('../../data/severstal/vae_train_labels.csv', '../../data/severstal/train_images')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = ConditionalVAE(latent_dim=128, num_classes=3).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

kl_weight = 0.01
num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    total_recon_loss = 0
    total_kl_loss = 0
    for images, cond in train_loader:
        images, cond = images.to(device), cond.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(images, cond)

        recon_loss = F.mse_loss(recon, images, reduction='sum') / images.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
        loss = recon_loss + kl_weight * kl_loss

        loss.backward()
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    avg_recon = total_recon_loss / len(train_loader)
    avg_kl = total_kl_loss / len(train_loader)
    print(f'epoch {epoch+1}/{num_epochs} - recon loss: {avg_recon:.2f} - kl loss: {avg_kl:.2f}')

torch.save(model.state_dict(), 'cvae.pth')
print('saved model to cvae.pth')