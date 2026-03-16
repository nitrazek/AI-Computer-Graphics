import os
import cv2
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, input_paths, target_paths, data_offset=0, data_size=None):
        input_files = sorted([f for f in os.listdir(input_paths) if f.endswith('.png')])
        target_files = sorted([f for f in os.listdir(target_paths) if f.endswith('.png')])

        if len(input_files) != len(target_files):
            raise ValueError('Input and target folders contain a different number of images.')
        if input_files != target_files:
            raise ValueError('Input and target filenames do not match after sorting.')

        end_index = None if data_size is None else data_offset + data_size
        selected_files = input_files[data_offset:end_index]

        self.input_paths = [os.path.join(input_paths, file_name) for file_name in selected_files]
        self.target_paths = [os.path.join(target_paths, file_name) for file_name in selected_files]

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