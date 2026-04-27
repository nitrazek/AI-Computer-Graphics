import torch
import numpy as np
import os
import re
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import hausdorff_distance
from skimage.morphology import remove_small_objects
import lpips
from flip_loss import LDRFLIPLoss
import csv

# Import your classes
from dataset import PhongDataset
from model import PhongGenerator


def get_latest_generator_checkpoint(checkpoints_dir="../checkpoints"):
    pattern = re.compile(r"^generator_(\d+)\.pth$")
    latest_mtime = -1.0
    latest_path = None
    for name in os.listdir(checkpoints_dir):
        if name == "generator_latest.pth":
            full_path = os.path.join(checkpoints_dir, name)
            mtime = os.path.getmtime(full_path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = full_path
            continue
        match = pattern.match(name)
        if match:
            full_path = os.path.join(checkpoints_dir, name)
            mtime = os.path.getmtime(full_path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = full_path
    if latest_path is None:
        raise FileNotFoundError(f"No generator checkpoint found in {checkpoints_dir}")
    return latest_path

def calculate_metrics():
    print("Starting Evaluation Phase...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load the trained Generator
    condition_dim = 10
    latent_dim = 100
    generator = PhongGenerator(condition_dim, latent_dim).to(device)
    
    model_path = get_latest_generator_checkpoint("../checkpoints")
    print(f"Using checkpoint: {model_path}")
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
            noise = torch.zeros(1, latent_dim, device=device)
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

    # 4. Print Final Report
    csv_path = "../results/evaluation_metrics.csv"
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Metric', 'Average Score', 'Better Score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Metric': 'FLIP', 'Average Score': f"{np.mean(flip_scores):.4f}", 'Better Score': 'Lower is better, Min 0.0'})
        writer.writerow({'Metric': 'LPIPS', 'Average Score': f"{np.mean(lpips_scores):.4f}", 'Better Score': 'Lower is better, Min 0.0'})
        writer.writerow({'Metric': 'SSIM', 'Average Score': f"{np.mean(ssim_scores):.4f}", 'Better Score': 'Higher is better, Max 1.0'})
        if hausdorff_scores:
            writer.writerow({'Metric': 'Hausdorff Distance', 'Average Score': f"{np.mean(hausdorff_scores):.4f}", 'Better Score': 'Lower is better, Min 0.0'})

if __name__ == '__main__':
    calculate_metrics()