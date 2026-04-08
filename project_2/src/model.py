import torch
import torch.nn as nn


class ExposureSynthesisCNN(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.underexposed_head = nn.Conv2d(features, in_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.overexposed_head = nn.Conv2d(features, in_channels, kernel_size=3, padding=1, padding_mode='reflect')

        nn.init.zeros_(self.underexposed_head.weight)
        nn.init.zeros_(self.underexposed_head.bias)
        nn.init.zeros_(self.overexposed_head.weight)
        nn.init.zeros_(self.overexposed_head.bias)

    def forward(self, x):
        features = self.encoder(x)
        underexposed = x + self.underexposed_head(features)
        overexposed = x + self.overexposed_head(features)
        return underexposed, overexposed