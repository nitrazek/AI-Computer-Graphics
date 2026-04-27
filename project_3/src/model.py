import torch
import torch.nn as nn

import torch
import torch.nn as nn

class PhongGenerator(nn.Module):
    def __init__(self, condition_dim=10, latent_dim=100):
        super(PhongGenerator, self).__init__()
        
        # Combine the random noise (latent_dim) and your JSON labels (condition_dim)
        self.fc = nn.Linear(condition_dim + latent_dim, 256 * 8 * 8)
        
        self.conv_blocks = nn.Sequential(
            # Input shape: [Batch, 256, 8, 8]
            
            # --- BLOCK 1: 8x8 -> 16x16 ---
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # --- BLOCK 2: 16x16 -> 32x32 ---
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # --- BLOCK 3: 32x32 -> 64x64 ---
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # --- BLOCK 4: 64x64 -> 128x128 ---
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),    
            
            # Output Shape: [Batch, 3, 128, 128]
            nn.Tanh() 
        )

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        return self.conv_blocks(x)
        
class PhongDiscriminator(nn.Module):
    def __init__(self, condition_dim=10):
        super(PhongDiscriminator, self).__init__()
        
        # We take the 3 image channels (RGB) + 10 condition channels
        self.model = nn.Sequential(
            # Input: [Batch, 13, 128, 128]
            nn.Conv2d(3 + condition_dim, 64, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            
            # Shape: [Batch, 64, 64, 64]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Shape: [Batch, 128, 32, 32]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Shape: [Batch, 256, 16, 16]
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Shape: [Batch, 512, 8, 8]
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid() # Outputs a percentage: 1.0 = 100% Real, 0.0 = 100% Fake
        )

    def forward(self, img, condition):
        # To combine 1D labels with a 2D image, we stretch the labels across the whole image size
        # Condition goes from [Batch, 10] -> [Batch, 10, 128, 128]
        cond_spatial = condition.view(condition.size(0), condition.size(1), 1, 1)
        cond_spatial = cond_spatial.expand(-1, -1, img.size(2), img.size(3))
        
        # Stick the image and the stretched labels together like a sandwich
        x = torch.cat([img, cond_spatial], dim=1)
        return self.model(x)
    
