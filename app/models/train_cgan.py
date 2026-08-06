import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from torchvision.utils import save_image
from dataset_gan import GANDataset
from cgan import ConditionalGenerator, ConditionalDiscriminator

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('using device:', device)

train_ds = GANDataset('../../data/severstal/vae_train_labels.csv', '../../data/severstal/train_images')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

latent_dim = 100
generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=3).to(device)
discriminator = ConditionalDiscriminator(num_classes=3).to(device)

g_optimizer = Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

num_epochs = 50

for epoch in range(num_epochs):
    total_g_loss = 0
    total_d_loss = 0

    for batch_idx, (real_images, cond) in enumerate(train_loader):
        real_images, cond = real_images.to(device), cond.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(z, cond)

        if batch_idx % 2 == 0:
            d_optimizer.zero_grad()
            real_preds = discriminator(real_images, cond)
            d_real_loss = criterion(real_preds, real_labels)
            fake_preds = discriminator(fake_images.detach(), cond)
            d_fake_loss = criterion(fake_preds, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            total_d_loss += d_loss.item()

        g_optimizer.zero_grad()
        fake_preds = discriminator(fake_images, cond)
        g_loss = criterion(fake_preds, real_labels)
        g_loss.backward()
        g_optimizer.step()
        total_g_loss += g_loss.item()

    avg_d_loss = total_d_loss / (len(train_loader) // 2)
    avg_g_loss = total_g_loss / len(train_loader)
    print(f'epoch {epoch+1}/{num_epochs} - d_loss: {avg_d_loss:.4f} - g_loss: {avg_g_loss:.4f}')

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            sample_cond = torch.eye(3).to(device)
            z = torch.randn(3, latent_dim, device=device)
            samples = generator(z, sample_cond)
            save_image(samples, f'cgan_samples_epoch{epoch+1}.png', normalize=True)

torch.save(generator.state_dict(), 'cgan_generator.pth')
torch.save(discriminator.state_dict(), 'cgan_discriminator.pth')
print('saved generator and discriminator')