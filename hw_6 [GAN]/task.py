# experiments

import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5, input_channels=3):
        super().__init__()
        modules = [nn.Conv2d(input_channels, start_channels, kernel_size=1, stride=1, padding=0), nn.ReLU()]
        for i in range(downsamplings):
            modules.append(nn.Conv2d(start_channels * 2 ** i, start_channels * 2 ** (i + 1), kernel_size=3, stride=2,
                                     padding=1))
            modules.append(nn.BatchNorm2d(start_channels * 2 ** (i + 1)))
            modules.append(nn.ReLU())
        modules.append(nn.Flatten())
        self.model = nn.Sequential(*modules)
        modules.append(nn.Linear(start_channels * img_size ** 2 // 2 ** downsamplings, 2 * latent_size))

    def forward(self, x):
        mu, log_sigma = torch.chunk(self.model(x), 2, dim=-1)
        sigma = torch.exp(log_sigma)
        z = mu + torch.randn_like(mu) * sigma
        return z, (mu, sigma)


class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5, output_channels=3):
        super().__init__()
        modules = [
            nn.Linear(latent_size, end_channels * img_size ** 2 // 2 ** upsamplings),
            nn.Unflatten(-1,
                         (end_channels * 2 ** upsamplings, img_size // 2 ** upsamplings, img_size // 2 ** upsamplings))
        ]

        for i in reversed(range(upsamplings)):
            modules.append(
                nn.ConvTranspose2d(end_channels * 2 ** (i + 1), end_channels * 2 ** i, kernel_size=4, stride=2,
                                   padding=1))
            modules.append(nn.BatchNorm2d(end_channels * 2 ** i))
            modules.append(nn.ReLU())
        modules.append(nn.Conv2d(end_channels, output_channels, kernel_size=1, stride=1, padding=0))
        if output_channels == 3:
            modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        return self.model(z)


class VAE(nn.Module):
    def __init__(self, input_channels=3, img_size=128, downsamplings=4, latent_size=256, down_channels=6,
                 up_channels=8):
        super().__init__()
        self.encoder = Encoder(img_size, latent_size, down_channels, downsamplings, input_channels)
        self.decoder = Decoder(img_size, latent_size, up_channels, downsamplings, input_channels)

    def forward(self, x):
        z, (mu, sigma) = self.encoder(x)
        x_pred = self.decoder(z)
        kld = 0.5 * (sigma ** 2 + mu ** 2 - torch.log(sigma ** 2) - 1)
        return x_pred, kld

    def encode(self, x):
        return self.encoder(x)[0]

    def encode_full(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def save(self):
        torch.save(self.state_dict(), "model.pth")

    def load(self):
        self.load_state_dict(torch.load(f'{__file__[:-7]}/model.pth'))
