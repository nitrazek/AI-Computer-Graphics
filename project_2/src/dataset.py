import os
import cv2
import torch
from torch.utils.data import Dataset

def _get_image_files(folder):
    valid_extensions = ('.png', '.tif', '.tiff')
    return sorted([file_name for file_name in os.listdir(folder) if file_name.lower().endswith(valid_extensions)])

class ImageDataset(Dataset):
    def __init__(self, ldr_paths, underexposed_paths, overexposed_paths):
        input_files = _get_image_files(ldr_paths)
        underexposed_files = _get_image_files(underexposed_paths)
        overexposed_files = _get_image_files(overexposed_paths)

        if len(input_files) != len(underexposed_files):
            raise ValueError('Input and underexposed folders contain a different number of images.')
        if input_files != underexposed_files:
            raise ValueError('Input and underexposed filenames do not match after sorting.')
        if len(input_files) != len(overexposed_files):
            raise ValueError('Input and overexposed folders contain a different number of images.')
        if input_files != overexposed_files:
            raise ValueError('Input and overexposed filenames do not match after sorting.')

        self.input_paths = [os.path.join(ldr_paths, file_name) for file_name in input_files]
        self.underexposed_paths = [os.path.join(underexposed_paths, file_name) for file_name in underexposed_files]
        self.overexposed_paths = [os.path.join(overexposed_paths, file_name) for file_name in overexposed_files]

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, key):
        input_image = cv2.imread(self.input_paths[key])
        underexposed_image = cv2.imread(self.underexposed_paths[key])
        overexposed_image = cv2.imread(self.overexposed_paths[key])
        
        if input_image is None:
            raise ValueError(f"Failed to load image: {self.input_paths[key]}")
        if underexposed_image is None:
            raise ValueError(f"Failed to load image: {self.underexposed_paths[key]}")
        if overexposed_image is None:
            raise ValueError(f"Failed to load image: {self.overexposed_paths[key]}")
        
        # BGR -> RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        underexposed_image = cv2.cvtColor(underexposed_image, cv2.COLOR_BGR2RGB)
        overexposed_image = cv2.cvtColor(overexposed_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        input_image = input_image.astype('float32') / 255.0
        underexposed_image = underexposed_image.astype('float32') / 255.0
        overexposed_image = overexposed_image.astype('float32') / 255.0

        # Transpose to (C, H, W)
        input_image = input_image.transpose(2, 0, 1)
        underexposed_image = underexposed_image.transpose(2, 0, 1)
        overexposed_image = overexposed_image.transpose(2, 0, 1)

        # Convert to torch tensors
        input_image = torch.from_numpy(input_image)
        underexposed_image = torch.from_numpy(underexposed_image)
        overexposed_image = torch.from_numpy(overexposed_image)
        
        return input_image, underexposed_image, overexposed_image
    
if __name__ == "__main__":
    dataset = ImageDataset(
        ldr_paths='../data/training/ldr',
        underexposed_paths='../data/training/underexposed',
        overexposed_paths='../data/training/overexposed'
    )
    print(f"Dataset size: {len(dataset)}")
    input_image, underexposed_image, overexposed_image = dataset[0]
    print(f"Input image shape: {input_image.shape}, Underexposed image shape: {underexposed_image.shape}, Overexposed image shape: {overexposed_image.shape}")