import os
import cv2
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, input_paths, target_paths, data_offset=0, data_size=200):
        self.input_paths = sorted([os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith('.png')])[data_offset : data_offset + data_size]
        self.target_paths = sorted([os.path.join(target_paths, f) for f in os.listdir(target_paths) if f.endswith('.png')])[data_offset : data_offset + data_size]

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, key):
        input_image = cv2.imread(self.input_paths[key])
        target_image = cv2.imread(self.target_paths[key])
        
        if input_image is None:
            raise ValueError(f"Failed to load image: {self.input_paths[key]}")
        if target_image is None:
            raise ValueError(f"Failed to load image: {self.target_paths[key]}")
        
        # BGR -> RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        input_image = input_image.astype('float32') / 255.0
        target_image = target_image.astype('float32') / 255.0

        # Transpose to (C, H, W)
        input_image = input_image.transpose(2, 0, 1)
        target_image = target_image.transpose(2, 0, 1)

        # Convert to torch tensors
        input_image = torch.from_numpy(input_image)
        target_image = torch.from_numpy(target_image)
        
        return input_image, target_image