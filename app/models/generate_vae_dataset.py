import torch
from torchvision.utils import save_image
from cvae import ConditionalVAE
import pandas as pd
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = ConditionalVAE(latent_dim=128, num_classes=3).to(device)
model.load_state_dict(torch.load('cvae.pth', map_location=device))
model.eval()

out_dir = '../../data/severstal/vae_synthetic'
os.makedirs(out_dir, exist_ok=True)

class_conditions = {
    'class_1': torch.tensor([1., 0., 0.]),
    'class_2': torch.tensor([0., 1., 0.]),
    'class_4': torch.tensor([0., 0., 1.]),
}

n_per_class = 1500
rows = []

with torch.no_grad():
    for class_name, cond in class_conditions.items():
        cond_batch = cond.unsqueeze(0).repeat(n_per_class, 1).to(device)
        z = torch.randn(n_per_class, 128).to(device)
        samples = model.decoder(z, cond_batch).cpu()

        for i in range(n_per_class):
            fname = f'{class_name}_synthetic_{i}.png'
            save_image(samples[i], os.path.join(out_dir, fname))
            rows.append({
                'ImageId': fname,
                'class_1': 1 if class_name == 'class_1' else 0,
                'class_2': 1 if class_name == 'class_2' else 0,
                'class_3': 0,
                'class_4': 1 if class_name == 'class_4' else 0,
            })

synthetic_labels = pd.DataFrame(rows)
synthetic_labels.to_csv('../../data/severstal/vae_synthetic_labels.csv', index=False)
print('generated', len(rows), 'synthetic images')