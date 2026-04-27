import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Import your classes
from dataset import PhongDataset
from model import PhongGenerator

def generate_individual_pairs():
    print("Generating individual comparison pairs...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load the trained Generator
    condition_dim = 10
    latent_dim = 100
    generator = PhongGenerator(condition_dim, latent_dim).to(device)
    
    checkpoint_path = "../checkpoints/generator_50.pth" 
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    generator.eval() 

    # 2. Load the Test Dataset
    dataset = PhongDataset(data_dir="../output/test", is_train=False)
    
    # Grab the first 12 images
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False) 
    conditions, real_imgs = next(iter(dataloader))
    
    conditions = conditions.to(device)
    real_imgs = real_imgs.to(device)
    
    # 3. Generate the Fake Images
    with torch.no_grad():
        noise = torch.randn(12, latent_dim, device=device)
        fake_imgs = generator(noise, conditions)
        
    # 4. Format images for saving (Shift from [-1.0, 1.0] back to [0.0, 1.0])
    real_imgs_01 = (real_imgs * 0.5) + 0.5
    fake_imgs_01 = (fake_imgs * 0.5) + 0.5
    
    # 5. Create a clean folder for the outputs
    output_dir = "../results/comparison_pairs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. Loop through and save them one by one
    for i in range(12):
        single_real = real_imgs_01[i]
        single_fake = fake_imgs_01[i]
        
        # Concatenate along dimension 2 (the width dimension) for a side-by-side image
        # If you prefer top-and-bottom, change 'dim=2' to 'dim=1'
        pair_tensor = torch.cat((single_real, single_fake), dim=2)
        
        filename = f"{output_dir}/pair_{i:02d}.png"
        save_image(pair_tensor, filename)
        
    print(f"Success! 12 pairs saved to the '{output_dir}' folder.")

if __name__ == '__main__':
    generate_individual_pairs()