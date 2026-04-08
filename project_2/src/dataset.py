import csv
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_path(data_root, relative_path):
    return os.path.normpath(os.path.join(data_root, relative_path))


def load_image_rgb01(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Failed to load image: {image_path}')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image)


def random_crop(input_image, underexposed_target, overexposed_target, crop_size):
    _, height, width = input_image.shape
    if height < crop_size or width < crop_size:
        raise ValueError('Crop size must be smaller than image dimensions.')

    top = np.random.randint(0, height - crop_size + 1)
    left = np.random.randint(0, width - crop_size + 1)
    bottom = top + crop_size
    right = left + crop_size

    return (
        input_image[:, top:bottom, left:right],
        underexposed_target[:, top:bottom, left:right],
        overexposed_target[:, top:bottom, left:right],
    )


class ExposureDataset(Dataset):
    def __init__(self, metadata_path, data_root='../data', data_offset=0, data_size=None, crop_size=None, patches_per_image=1):
        with open(metadata_path, newline='') as metadata_file:
            rows = list(csv.DictReader(metadata_file))

        end_index = None if data_size is None else data_offset + data_size
        selected_rows = rows[data_offset:end_index]

        self.crop_size = crop_size
        self.patches_per_image = patches_per_image if crop_size is not None else 1
        self.samples = []
        for row in selected_rows:
            self.samples.append({
                'scene_name': row['scene_name'],
                'input_path': resolve_path(data_root, row['ldr_path']),
                'underexposed_path': resolve_path(data_root, row['underexposed_path']),
                'overexposed_path': resolve_path(data_root, row['overexposed_path']),
                'reference_exposure_time': float(row['reference_exposure_time']),
                'underexposed_exposure_time': float(row['underexposed_exposure_time']),
                'overexposed_exposure_time': float(row['overexposed_exposure_time']),
            })

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, key):
        sample = self.samples[key % len(self.samples)]
        input_image = load_image_rgb01(sample['input_path'])
        underexposed_target = load_image_rgb01(sample['underexposed_path'])
        overexposed_target = load_image_rgb01(sample['overexposed_path'])

        if self.crop_size is not None:
            input_image, underexposed_target, overexposed_target = random_crop(
                input_image,
                underexposed_target,
                overexposed_target,
                self.crop_size,
            )

        metadata = {
            'scene_name': sample['scene_name'],
            'reference_exposure_time': torch.tensor(sample['reference_exposure_time'], dtype=torch.float32),
            'underexposed_exposure_time': torch.tensor(sample['underexposed_exposure_time'], dtype=torch.float32),
            'overexposed_exposure_time': torch.tensor(sample['overexposed_exposure_time'], dtype=torch.float32),
        }

        return input_image, underexposed_target, overexposed_target, metadata