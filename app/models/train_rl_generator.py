import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from dataset_gan import GANDataset
from cgan import ConditionalGenerator, ConditionalDiscriminator
from classifier import DefectCNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('using device:', device)

train_ds = GANDataset('../../data/severstal/vae_train_labels.csv', '../../data/severstal/train_images')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

latent_dim = 100

generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=3).to(device)
generator.load_state_dict(torch.load('cgan_generator.pth', map_location=device))

discriminator = ConditionalDiscriminator(num_classes=3).to(device)
discriminator.load_state_dict(torch.load('cgan_discriminator.pth', map_location=device))

reward_model = DefectCNN().to(device)
reward_model.load_state_dict(torch.load('gan_oversampled_cnn.pth', map_location=device))
reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False

g_optimizer = Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
adv_criterion = nn.BCEWithLogitsLoss()

reward_weight = 0.5

# class index mapping: 0 -> class_1, 1 -> class_2, 2 -> class_4
# these correspond to positions in the reward_model's 4-output vector: [class_1, class_2, class_3, class_4]
reward_output_index = {0: 0, 1: 1, 2: 3}

num_epochs = 20

for epoch in range(num_epochs):
    total_adv_loss = 0
    total_reward_loss = 0
    total_reward = 0

    for batch_idx, (real_images, cond) in enumerate(train_loader):
        real_images, cond = real_images.to(device), cond.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        z = torch.randn(batch_size, latent_dim, device=device)
        generated_images = generator(z, cond)

        if batch_idx % 2 == 0:
            d_optimizer.zero_grad()
            real_preds = discriminator(real_images, cond)
            d_real_loss = adv_criterion(real_preds, real_labels)
            fake_preds = discriminator(generated_images.detach(), cond)
            d_fake_loss = adv_criterion(fake_preds, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

        g_optimizer.zero_grad()

        fake_preds = discriminator(generated_images, cond)
        adv_loss = adv_criterion(fake_preds, real_labels)

        rescaled_images = (generated_images + 1) / 2
        resized_images = F.interpolate(rescaled_images, size=(256, 256), mode='bilinear', align_corners=False)
        reward_logits = reward_model(resized_images)
        reward_probs = torch.sigmoid(reward_logits)

        class_indices = cond.argmax(dim=1)
        target_cols = torch.tensor(
            [reward_output_index[idx.item()] for idx in class_indices], device=device
        )
        target_rewards = reward_probs[torch.arange(batch_size), target_cols]

        reward_loss = -target_rewards.mean()

        combined_loss = adv_loss + reward_weight * reward_loss
        combined_loss.backward()
        g_optimizer.step()

        total_adv_loss += adv_loss.item()
        total_reward_loss += reward_loss.item()
        total_reward += target_rewards.mean().item()

    avg_adv_loss = total_adv_loss / len(train_loader)
    avg_reward_loss = total_reward_loss / len(train_loader)
    avg_reward = total_reward / len(train_loader)
    print(f'epoch {epoch+1}/{num_epochs} - adv_loss: {avg_adv_loss:.4f} - reward_loss: {avg_reward_loss:.4f} - avg_reward: {avg_reward:.4f}')

torch.save(generator.state_dict(), 'rl_generator.pth')
print('saved RL-tuned generator to rl_generator.pth')