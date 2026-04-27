import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Import your classes
from dataset import PhongDataset
from model import PhongGenerator, PhongDiscriminator

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs("../results/epoch_images", exist_ok=True)
    os.makedirs("../checkpoints", exist_ok=True)

    # Hyperparameters
    batch_size = 64
    epochs = 50       
    latent_dim = 100  
    condition_dim = 10 

    # Load Data
    dataset = PhongDataset(data_dir="../output/train", is_train=True) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Models (Make sure you updated model.py with the Resize-Conv code!)
    generator = PhongGenerator(condition_dim, latent_dim).to(device)
    discriminator = PhongDiscriminator(condition_dim).to(device)

    # Loss Functions
    criterion_BCE = nn.BCELoss() # Adversarial Loss
    criterion_L1 = nn.L1Loss() # Pixel Matching Loss
    lambda_L1 = 0.0 # Weight for L1 Loss
    
    # Optimizers (TTUR: Discriminator learns at half speed)
    opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, latent_dim, device=device)
    fixed_conditions, _ = next(iter(dataloader))
    fixed_conditions = fixed_conditions[:16].to(device)

    # Track the best score
    best_g_loss = float('inf')

    print("Starting Training Loop...")
    for epoch in range(epochs):
        
        # Track the total Generator loss for this specific epoch
        epoch_g_loss = 0.0 

        for i, (conditions, real_imgs) in enumerate(dataloader):
            
            conditions = conditions.to(device)
            real_imgs = real_imgs.to(device)
            curr_batch_size = real_imgs.size(0)

            # Label Smoothing (Real = 0.9, Fake = 0.1)
            real_labels = torch.ones(curr_batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(curr_batch_size, 1, device=device) * 0.1

            # ==========================================
            # Train Discriminator
            # ==========================================
            opt_D.zero_grad() 

            outputs_real = discriminator(real_imgs, conditions)
            loss_D_real = criterion_BCE(outputs_real, real_labels)

            noise = torch.randn(curr_batch_size, latent_dim, device=device)
            fake_imgs = generator(noise, conditions)

            outputs_fake = discriminator(fake_imgs.detach(), conditions)
            loss_D_fake = criterion_BCE(outputs_fake, fake_labels)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            opt_D.step()

            # ==========================================
            # Train Generator
            # ==========================================
            opt_G.zero_grad()

            # 1. Trick the Discriminator
            outputs = discriminator(fake_imgs, conditions)
            loss_G_bce = criterion_BCE(outputs, real_labels) 

            # 2. Pixel-Perfect matching (L1)
            loss_G_L1 = criterion_L1(fake_imgs, real_imgs) * lambda_L1

            # Combine and Update
            loss_G = loss_G_bce + loss_G_L1
            loss_G.backward()
            opt_G.step()

            # Add to the running total for the epoch
            epoch_g_loss += loss_G.item()

            if i % 10 == 0:
                print(f"[{epoch}/{epochs}][{i}/{len(dataloader)}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

        # ==========================================
        # End of Epoch: Save Images & Evaluate Best Model
        # ==========================================
        
        # 1. Save Test Grid
        with torch.no_grad(): 
            test_fakes = generator(fixed_noise, fixed_conditions)
            test_fakes = (test_fakes * 0.5) + 0.5 
            save_image(test_fakes, f"../results/epoch_images/epoch_{epoch:03}.png", nrow=4)

        # 2. Save all models
        torch.save(generator.state_dict(), f"../checkpoints/generator_{epoch + 1}.pth")
        torch.save(discriminator.state_dict(), f"../checkpoints/discriminator_{epoch + 1}.pth")

    print("Training Complete!")

if __name__ == '__main__':
    train()