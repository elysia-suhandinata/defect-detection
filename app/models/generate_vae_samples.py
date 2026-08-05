import torch
import matplotlib.pyplot as plt
from cvae import ConditionalVAE

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = ConditionalVAE(latent_dim=128, num_classes=3).to(device)
model.load_state_dict(torch.load('cvae.pth', map_location=device))
model.eval()

class_names = ['class_1', 'class_2', 'class_4']
conditions = torch.eye(3)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

with torch.no_grad():
    for row, cond in enumerate(conditions):
        cond_batch = cond.unsqueeze(0).repeat(4, 1).to(device)
        z = torch.randn(4, 128).to(device)
        samples = model.decoder(z, cond_batch).cpu()

        for col in range(4):
            img = samples[col].permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(class_names[row])

    for row in range(3):
        axes[row, 0].text(-30, 128, class_names[row], rotation=90, va='center')

plt.tight_layout()
plt.savefig('cvae_samples.png')
print('saved samples to cvae_samples.png')