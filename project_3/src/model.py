import torch
import torch.nn as nn


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class PhongGenerator(nn.Module):
    def __init__(self, condition_dim=10, latent_dim=100):
        super(PhongGenerator, self).__init__()

        # Combine conditioning values with latent code.
        self.fc = nn.Linear(condition_dim + latent_dim, 256 * 8 * 8)

        # Resize-conv blocks are less prone to checkerboard artifacts than transposed conv.
        self.upsample_blocks = nn.Sequential(
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.to_mask = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, noise, condition, return_aux=False):
        x = torch.cat([noise, condition], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        features = self.upsample_blocks(x)

        rgb = self.to_rgb(features)
        mask = self.to_mask(features)

        rgb_01 = (rgb * 0.5) + 0.5
        composite_01 = rgb_01 * mask
        composite = (composite_01 * 2.0) - 1.0

        if return_aux:
            return composite, rgb, mask
        return composite


class PhongDiscriminator(nn.Module):
    def __init__(self, condition_dim=10):
        super(PhongDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3 + condition_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),
        )

    def forward(self, img, condition):
        cond_spatial = condition.view(condition.size(0), condition.size(1), 1, 1)
        cond_spatial = cond_spatial.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, cond_spatial], dim=1)
        return self.model(x)
