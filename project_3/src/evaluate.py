import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import hausdorff_distance
from skimage.morphology import remove_small_objects
import lpips
from flip_loss import LDRFLIPLoss

# Import your classes
from dataset import PhongDataset
from model import PhongGenerator

def calculate_metrics():
    print("Starting Evaluation Phase...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load the trained Generator
    condition_dim = 10
    latent_dim = 100
    generator = PhongGenerator(condition_dim, latent_dim).to(device)
    
    model_path = "../checkpoints/generator_50.pth" 
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    generator.eval() # Set to evaluation mode (turns off batchnorm/dropout)

    # 2. Load the Dataset
    dataset = PhongDataset(data_dir="../output/test", is_train=False)
    # Batch size 1 is easier for image-by-image evaluation
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 


    flip_scores = []
    flip_calculator = LDRFLIPLoss().to(device)
    lpips_scores = []
    lpips_calculator = lpips.LPIPS(net='alex').to(device)
    ssim_scores = []
    hausdorff_scores = []
    
    with torch.no_grad():
        for (condition, real_img) in dataloader:
                
            condition = condition.to(device)
            real_img = real_img.to(device)
            
            # Generate the fake image
            noise = torch.randn(1, latent_dim, device=device)
            fake_img = generator(noise, condition)
            
            # 3. Format images for the SSIM and Hausdorff calculations
            # image range: [0.0, 1.0]
            fake_img_np = ((fake_img[0].cpu().numpy().transpose(1, 2, 0)) * 0.5) + 0.5
            real_img_np = ((real_img[0].cpu().numpy().transpose(1, 2, 0)) * 0.5) + 0.5
            
            # -- CALCULATE FLIP --
            # image range for FLIP: [0.0, 1.0]
            fake_img_01 = (fake_img * 0.5) + 0.5
            real_img_01 = (real_img * 0.5) + 0.5
            current_flip = flip_calculator(fake_img_01, real_img_01).mean().item()
            flip_scores.append(current_flip)

            # -- CALCULATE LPIPS --
            # image range for LPIPS: [-1.0, 1.0]
            current_lpips = lpips_calculator(fake_img, real_img).item()
            lpips_scores.append(current_lpips)

            # -- CALCULATE SSIM --
            # image range for SSIM: [0.0, 1.0]
            current_ssim = ssim(real_img_np, fake_img_np, channel_axis=-1, data_range=1.0)
            ssim_scores.append(current_ssim)

            # -- CALCULATE HAUSDORFF DISTANCE --
            fake_gray = np.mean(fake_img_np, axis=2)
            real_gray = np.mean(real_img_np, axis=2)
            fake_binary = remove_small_objects(fake_gray > 0.02)
            real_binary = remove_small_objects(real_gray > 0.02)
            
            if np.any(fake_binary) and np.any(real_binary):
                current_hausdorff = hausdorff_distance(real_binary, fake_binary)
                hausdorff_scores.append(current_hausdorff)

            # Print progress

    # 4. Print Final Report
    print("\n" + "="*40)
    print("Average FLIP:      {:.4f} (Lower is better, Min 0.0)".format(np.mean(flip_scores)))
    print("Average LPIPS:     {:.4f} (Lower is better, Min 0.0)".format(np.mean(lpips_scores)))
    # SSIM ranges from -1.0 to 1.0 (1.0 is a perfect identical match)
    print(f"Average SSIM:      {np.mean(ssim_scores):.4f} (Higher is better, Max 1.0)")
    # Hausdorff is a physical pixel distance (0.0 means the shapes perfectly overlap)
    print(f"Average Hausdorff: {np.mean(hausdorff_scores):.4f} (Lower is better, Min 0.0)")
    print("="*40)

if __name__ == '__main__':
    calculate_metrics()