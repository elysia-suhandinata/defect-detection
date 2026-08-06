import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=3, embed_dim=32):
        super().__init__()
        self.cond_embed = nn.Linear(num_classes, embed_dim)
        self.fc = nn.Linear(latent_dim + embed_dim, 512 * 4 * 4)

        self.net = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, cond):
        cond_emb = self.cond_embed(cond)
        x = torch.cat([z, cond_emb], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.net(x)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 + num_classes, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x, cond):
        cond_map = cond.view(cond.size(0), cond.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond_map], dim=1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)