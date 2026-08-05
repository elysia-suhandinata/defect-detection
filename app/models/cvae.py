import torch
import torch.nn as nn

def swish(x):
    return x * torch.sigmoid(x)

class ConditionalEncoder(nn.Module):
    def __init__(self, latent_dim=128, num_classes=3, embed_dim=32):
        super().__init__()
        self.cond_embed = nn.Linear(num_classes, embed_dim)
        self.conv1 = nn.Conv2d(3 + num_classes, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.fc_mu = nn.Linear(256 * 8 * 8 + embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8 + embed_dim, latent_dim)

    def forward(self, x, cond):
        cond_map = cond.view(cond.size(0), cond.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond_map], dim=1)
        cond_emb = swish(self.cond_embed(cond))
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, cond_emb], dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_classes=3, embed_dim=32):
        super().__init__()
        self.cond_embed = nn.Linear(num_classes, embed_dim)
        self.fc = nn.Linear(latent_dim + embed_dim, 256 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

    def forward(self, z, cond):
        cond_emb = swish(self.cond_embed(cond))
        x = torch.cat([z, cond_emb], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8)
        x = swish(self.deconv1(x))
        x = swish(self.deconv2(x))
        x = swish(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=3, embed_dim=32):
        super().__init__()
        self.encoder = ConditionalEncoder(latent_dim, num_classes, embed_dim)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, embed_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond)
        return recon, mu, logvar