import torch
import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

from dataset import ImageDataset
from model import ImageRestorationCNN

def evaluate(input_path, target_path, model_path, batch_size=1, result_csv='result.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ImageDataset(input_paths=input_path, target_paths=target_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ImageRestorationCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    psnr_values, ssim_values, lpips_values = [], [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # since the model output layer does not use sigmoid as activation function, the output values may be outside the [0, 1] range.
            outputs_clipped = torch.clamp(outputs, 0.0, 1.0)


if __name__ == "__main__":
    pass