import torch
from torchvision.utils import save_image
from cgan import ConditionalGenerator
import pandas as pd
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

latent_dim = 100
generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=3).to(device)
generator.load_state_dict(torch.load('cgan_generator.pth', map_location=device))
generator.eval()

out_dir = '../../data/severstal/gan_synthetic'
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
        z = torch.randn(n_per_class, latent_dim).to(device)
        samples = generator(z, cond_batch).cpu()

        for i in range(n_per_class):
            fname = f'{class_name}_gan_synthetic_{i}.png'
            save_image(samples[i], os.path.join(out_dir, fname), normalize=True)
            rows.append({
                'ImageId': fname,
                'class_1': 1 if class_name == 'class_1' else 0,
                'class_2': 1 if class_name == 'class_2' else 0,
                'class_3': 0,
                'class_4': 1 if class_name == 'class_4' else 0,
            })

synthetic_labels = pd.DataFrame(rows)
synthetic_labels.to_csv('../../data/severstal/gan_synthetic_labels.csv', index=False)
print('generated', len(rows), 'synthetic images')