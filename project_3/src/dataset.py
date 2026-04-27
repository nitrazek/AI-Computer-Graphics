import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PhongDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        # transforms.ToTensor() converts the image from [0, 255] pixels to [0.0, 1.0] float tensors
        # It also changes the shape from (Height, Width, Channels) to (Channels, Height, Width)
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # Count how many json files exist in the folder
        self.num_samples = len([f for f in os.listdir(data_dir) if f.endswith('.json')])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Load JSON labels
        if not self.is_train:
            idx += 2400
        json_path = os.path.join(self.data_dir, f'image_{idx:04}.json')
        with open(json_path, 'r') as f:
            labels = json.load(f)

        # 2. Extract the changing features to feed into the neural network
        # We extract the 10 randomized values: position (3), diffuse color (3), shininess (1), light position (3)
        # model_translation was [-5, 5] -> divide by 5
        norm_model = [x / 5.0 for x in labels["model_translation"]]
        
        # material_diffuse was [0.0, 1.0] -> multiply by 2 and subtract 1
        norm_diffuse = [(x * 2.0) - 1.0 for x in labels["material_diffuse"]]
        
        # shininess was [3, 20] -> rough normalization
        norm_shine = [(labels["material_shininess"] - 11.5) / 8.5]
        
        # light_position was [-20, 20] -> divide by 20
        norm_light = [x / 20.0 for x in labels["light_position"]]

        features = norm_model + norm_diffuse + norm_shine + norm_light
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # 3. Load the corresponding image
        img_path = os.path.join(self.data_dir, f'image_{idx:04}.png')
        image = Image.open(img_path).convert('RGB') # Convert to RGB to drop the Alpha (transparency) channel
        image_tensor = self.transform(image)

        return feature_tensor, image_tensor

if __name__ == "__main__":
    dataset = PhongDataset(data_dir="../output") 
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    features, images = next(iter(dataloader))
    print(f"Features shape: {features.shape}") # Should be [16, 10] (16 images, 10 parameters each)
    print(f"Images shape: {images.shape}")     # Should be [16, 3, 128, 128]