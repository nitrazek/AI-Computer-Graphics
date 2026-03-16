import torch
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

from dataset import ImageDataset
from model import ImageRestorationCNN

def evaluate(dataloader, model_path, result_csv='result.csv'):
    print(f"Evaluating model: {model_path}")
    print(f"Size of test set: {len(dataloader.dataset)}")
    print(f"Saving results to: {result_csv}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageRestorationCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    psnr_values, ssim_values, lpips_values = [], [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Since the model output layer does not use sigmoid as activation function, the output values may be outside the [0, 1] range.
            outputs_clipped = torch.clamp(outputs, 0.0, 1.0)

            # Convert tensors to numpy arrays for metric calculations
            # Pytorch tensor shape: (Batch, Channels, Height, Width), but PSNR and SSIM expect (Height, Width, Channels)
            outputs_np = outputs_clipped.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            targets_np = targets.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

            # Process all images in the batch
            for i in range(outputs_np.shape[0]):
                psnr_values.append(psnr(targets_np[i], outputs_np[i], data_range=1.0))
                ssim_values.append(ssim(targets_np[i], outputs_np[i], channel_axis=2, data_range=1.0, win_size=5))

            # For LPIPS:
            # 1. We need to convert the images back to the range [-1, 1]
            # 2. LPIPS is able to handle batched inputs
            outputs_lpips = outputs_clipped * 2 - 1  # [0, 1] -> [-1, 1]
            targets_lpips = targets * 2 - 1  # [0, 1] -> [-1, 1]
            lpips_batch = lpips_loss_fn(outputs_lpips, targets_lpips).cpu().numpy()
            # Extract individual values from the batch
            for j in range(lpips_batch.shape[0]):
                lpips_values.append(lpips_batch[j].item())
    
    print(f"Average PSNR: {np.mean(psnr_values):.4f}")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_values):.4f}")

    with open(result_csv, 'w') as f:
        f.write('PSNR, SSIM, LPIPS\n')
        for psnr_val, ssim_val, lpips_val in zip(psnr_values, ssim_values, lpips_values):
            f.write(f"{psnr_val:.4f}, {ssim_val:.4f}, {lpips_val:.4f}\n")


if __name__ == "__main__":
    dataset = ImageDataset(input_paths='../data/validation/noisy_001', target_paths='../data/validation/clean', data_offset=201, data_size=50)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    evaluate(dataloader, '../model/denoising_model.pth', result_csv='../results/denoising_results.csv')
    evaluate(dataloader, '../model/deblurring_model.pth', result_csv='../results/deblurring_results.csv')