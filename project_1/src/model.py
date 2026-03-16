import torch.nn as nn


class ImageRestorationCNN(nn.Module):
    def __init__(self, in_channels=3, features=64, num_blocks=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
