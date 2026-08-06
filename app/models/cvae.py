import torch
import torch.nn as nn

latent_dim = 128
num_classes = 3
embed_dim = 32
img_size = 128


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim):
        super().__init__()
        self.cond_embed = nn.Linear(num_classes, embed_dim)
        self.fc = nn.Linear(latent_dim + embed_dim, 256 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

    def forward(self, z, cond):
        cond_emb = self.cond_embed(cond)
        x = torch.cat([z, cond_emb], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = torch.relu(self.bn1(self.deconv1(x)))
        x = torch.relu(self.bn2(self.deconv2(x)))
        x = torch.relu(self.bn3(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3 + num_classes, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 8 * 8, 1)

    def forward(self, x, cond):
        cond_map = cond.view(cond.size(0), cond.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond_map], dim=1)
        x = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = x.view(x.size(0), -1)
        return self.fc(x)