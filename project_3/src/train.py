import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Import your classes
from dataset import PhongDataset
from model import PhongGenerator, PhongDiscriminator


def extract_real_mask(real_imgs, threshold=0.01):
    real_01 = (real_imgs * 0.5) + 0.5
    intensity = real_01.max(dim=1, keepdim=True).values
    mask = (intensity > threshold).float()
    # Slight dilation covers dim sphere edges that can be near-black.
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def weighted_l1_loss(fake_imgs, real_imgs, fg_threshold=0.02, fg_weight=10.0):
    real_01 = (real_imgs * 0.5) + 0.5
    fake_01 = (fake_imgs * 0.5) + 0.5
    intensity = real_01.max(dim=1, keepdim=True).values
    fg_mask = (intensity > fg_threshold).float()
    fg_mask = F.max_pool2d(fg_mask, kernel_size=3, stride=1, padding=1)
    weights = 1.0 + fg_mask * (fg_weight - 1.0)
    return torch.mean(torch.abs(fake_01 - real_01) * weights)

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
    lambda_adv = 0.25
    lambda_l1 = 35.0
    lambda_mask = 8.0
    lambda_bg = 30.0

    # Load Data
    dataset = PhongDataset(data_dir="../output/train", is_train=True) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Models (Make sure you updated model.py with the Resize-Conv code!)
    generator = PhongGenerator(condition_dim, latent_dim).to(device)
    discriminator = PhongDiscriminator(condition_dim).to(device)

    # Loss Functions
    criterion_adv = nn.BCEWithLogitsLoss()  # Adversarial loss on discriminator logits
    criterion_mask = nn.BCELoss()  # Supervise object silhouette
    
    # Optimizers (TTUR: Discriminator learns at half speed)
    opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    fixed_noise = torch.zeros(16, latent_dim, device=device)
    fixed_conditions, _ = next(iter(dataloader))
    fixed_conditions = fixed_conditions[:16].to(device)

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
            fake_labels = torch.zeros(curr_batch_size, 1, device=device)

            # ==========================================
            # Train Discriminator
            # ==========================================
            opt_D.zero_grad() 

            outputs_real = discriminator(real_imgs, conditions)
            loss_D_real = criterion_adv(outputs_real, real_labels)

            noise = torch.zeros(curr_batch_size, latent_dim, device=device)
            fake_imgs = generator(noise, conditions)

            outputs_fake = discriminator(fake_imgs.detach(), conditions)
            loss_D_fake = criterion_adv(outputs_fake, fake_labels)

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_D.step()

            # ==========================================
            # Train Generator
            # ==========================================
            opt_G.zero_grad()

            # 1. Trick the Discriminator
            outputs = discriminator(fake_imgs, conditions)
            loss_G_adv = criterion_adv(outputs, real_labels)

            # 2. Pixel-Perfect matching (L1)
            loss_G_L1 = weighted_l1_loss(fake_imgs, real_imgs)

            # 3. Force clean object boundaries and suppress background artifacts.
            _, _, fake_mask = generator(noise, conditions, return_aux=True)
            real_mask = extract_real_mask(real_imgs)
            loss_G_mask = criterion_mask(fake_mask, real_mask)

            fake_01 = (fake_imgs * 0.5) + 0.5
            bg_mask = 1.0 - real_mask
            loss_G_bg = torch.mean(torch.abs(fake_01 * bg_mask))

            # Combine and Update
            loss_G = (
                loss_G_adv * lambda_adv
                + loss_G_L1 * lambda_l1
                + loss_G_mask * lambda_mask
                + loss_G_bg * lambda_bg
            )
            loss_G.backward()
            opt_G.step()

            # Add to the running total for the epoch
            epoch_g_loss += loss_G.item()

            if i % 10 == 0:
                print(f"[{epoch + 1}/{epochs}][{i + 1}/{len(dataloader)}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

        # ==========================================
        # End of Epoch: Save Images & Evaluate Best Model
        # ==========================================
        
        # 1. Save Test Grid
        with torch.no_grad(): 
            test_fakes = generator(fixed_noise, fixed_conditions)
            test_fakes = (test_fakes * 0.5) + 0.5 
            save_image(test_fakes, f"../results/epoch_images/epoch_{epoch + 1:03}.png", nrow=4)

        # 2. Save all models
        torch.save(generator.state_dict(), f"../checkpoints/generator_{epoch + 1}.pth")
        torch.save(discriminator.state_dict(), f"../checkpoints/discriminator_{epoch + 1}.pth")
        torch.save(generator.state_dict(), "../checkpoints/generator_latest.pth")
        torch.save(discriminator.state_dict(), "../checkpoints/discriminator_latest.pth")

    print("Training Complete!")

if __name__ == '__main__':
    train()