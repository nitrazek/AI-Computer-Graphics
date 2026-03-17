import os
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_bilateral, richardson_lucy
import lpips
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import ImageRestorationCNN


def write_results(result_csv, psnr_values, ssim_values, lpips_values):
    average_psnr = float(np.mean(psnr_values))
    average_ssim = float(np.mean(ssim_values))
    average_lpips = float(np.mean(lpips_values))
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)

    with open(result_csv, 'w') as f:
        f.write('sample,psnr,ssim,lpips\n')
        f.write(f"average,{average_psnr:.4f},{average_ssim:.4f},{average_lpips:.4f}\n")
        for sample_index, (psnr_val, ssim_val, lpips_val) in enumerate(zip(psnr_values, ssim_values, lpips_values), start=1):
            f.write(f"{sample_index},{psnr_val:.4f},{ssim_val:.4f},{lpips_val:.4f}\n")

    return average_psnr, average_ssim, average_lpips


def build_dataloader(input_paths, target_paths, data_offset=0, data_size=50, batch_size=1):
    dataset = ImageDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        data_offset=data_offset,
        data_size=data_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate(dataloader, model_path, result_csv='result.csv'):
    print('=' * 50)
    print(f"Evaluating model: {model_path}")
    print(f"Size of test set: {len(dataloader.dataset)}")
    print(f"Saving results to: {result_csv}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageRestorationCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Initializing LPIPS...")
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    print("LPIPS initialized.")
    psnr_values, ssim_values, lpips_values = [], [], []

    with torch.no_grad():
        total_batches = len(dataloader)
        for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
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

            if batch_index % 5 == 0 or batch_index == total_batches:
                print(f"Progress: {batch_index}/{total_batches} batches ({(100.0 * batch_index / total_batches):.1f}%)")
    
    average_psnr, average_ssim, average_lpips = write_results(result_csv, psnr_values, ssim_values, lpips_values)

    print(f"Average PSNR: {average_psnr:.4f}")
    print(f"Average SSIM: {average_ssim:.4f}")
    print(f"Average LPIPS: {average_lpips:.4f}")
    print('=' * 50)

    return {
        'psnr': average_psnr,
        'ssim': average_ssim,
        'lpips': average_lpips,
    }

def evaluate_denoising_bilateral(dataloader, sigma_color=0.05, sigma_spatial=15, result_csv='bilateral_denoising_results.csv'):
    print('=' * 50)
    print(f"Evaluating bilateral denoising")
    print(f"Size of test set: {len(dataloader.dataset)}")
    print(f"Saving results to: {result_csv}")
    
    psnr_values, ssim_values, lpips_values = [], [], []
    print("Initializing LPIPS...")
    lpips_loss_fn = lpips.LPIPS(net='alex')
    print("LPIPS initialized.")

    total_batches = len(dataloader)
    for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
        inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        targets_np = targets.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        for i in range(inputs_np.shape[0]):
            restored = denoise_bilateral(inputs_np[i], sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=-1)
            restored = np.clip(restored, 0.0, 1.0)
            
            psnr_values.append(psnr(targets_np[i], restored, data_range=1.0))
            ssim_values.append(ssim(targets_np[i], restored, channel_axis=2, data_range=1.0, win_size=5))
            
            restored_lpips = torch.from_numpy(restored.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1  # [0, 1] -> [-1, 1]
            targets_lpips = torch.from_numpy(targets_np[i].transpose(2, 0, 1)).unsqueeze(0) * 2 - 1  # [0, 1] -> [-1, 1]
            lpips_value = lpips_loss_fn(restored_lpips.float(), targets_lpips.float()).item()
            lpips_values.append(lpips_value)

        if batch_index % 5 == 0 or batch_index == total_batches:
            print(f"Progress: {batch_index}/{total_batches} images ({(100.0 * batch_index / total_batches):.1f}%)")

    average_psnr, average_ssim, average_lpips = write_results(result_csv, psnr_values, ssim_values, lpips_values)

    print(f"Average PSNR: {average_psnr:.4f}")
    print(f"Average SSIM: {average_ssim:.4f}")
    print(f"Average LPIPS: {average_lpips:.4f}")
    print('=' * 50)

    return {
        'psnr': average_psnr,
        'ssim': average_ssim,
        'lpips': average_lpips,
    }

def evaluate_richardson_lucy(dataloader, psf, iterations=15, result_csv='richardson_lucy_deblurring_results.csv'):
    print('=' * 50)
    print(f"Evaluating Richardson-Lucy deblurring")
    print(f"Size of test set: {len(dataloader.dataset)}")
    print(f"Saving results to: {result_csv}")
    
    psnr_values, ssim_values, lpips_values = [], [], []
    print("Initializing LPIPS...")
    lpips_loss_fn = lpips.LPIPS(net='alex')
    print("LPIPS initialized.")

    total_batches = len(dataloader)
    for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
        inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        targets_np = targets.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        for i in range(inputs_np.shape[0]):
            # skimage.richardson_lucy expects image and PSF to have matching dimensionality.
            # For RGB images and a 2D PSF, deconvolve each channel separately.
            restored = np.stack([
                richardson_lucy(inputs_np[i][..., c], psf=psf, num_iter=iterations)
                for c in range(inputs_np[i].shape[2])
            ], axis=2)
            restored = np.clip(restored, 0.0, 1.0)
            
            psnr_values.append(psnr(targets_np[i], restored, data_range=1.0))
            ssim_values.append(ssim(targets_np[i], restored, channel_axis=2, data_range=1.0, win_size=5))
            
            restored_lpips = torch.from_numpy(restored.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1  # [0, 1] -> [-1, 1]
            targets_lpips = torch.from_numpy(targets_np[i].transpose(2, 0, 1)).unsqueeze(0) * 2 - 1  # [0, 1] -> [-1, 1]
            lpips_value = lpips_loss_fn(restored_lpips.float(), targets_lpips.float()).item()
            lpips_values.append(lpips_value)

        if batch_index % 5 == 0 or batch_index == total_batches:
            print(f"Progress: {batch_index}/{total_batches} images ({(100.0 * batch_index / total_batches):.1f}%)")

    average_psnr, average_ssim, average_lpips = write_results(result_csv, psnr_values, ssim_values, lpips_values)

    print(f"Average PSNR: {average_psnr:.4f}")
    print(f"Average SSIM: {average_ssim:.4f}")
    print(f"Average LPIPS: {average_lpips:.4f}")
    print('=' * 50)


if __name__ == "__main__":
    print('Evaluating denoising, with sigma=0.01')
    denoising_001_dataloader = build_dataloader('../data/validation/noisy_001', '../data/validation/clean', data_offset=100, data_size=50)
    evaluate(denoising_001_dataloader, '../models/denoising_001_model.pth', result_csv='../results/denoising_001_results.csv')
    evaluate_denoising_bilateral(denoising_001_dataloader, sigma_color=0.025, sigma_spatial=10, result_csv='../results/bilateral_denoising_001_results.csv')

    print('Evaluating denoising, with sigma=0.03')
    denoising_003_dataloader = build_dataloader('../data/validation/noisy_003', '../data/validation/clean', data_offset=100, data_size=50)
    evaluate(denoising_003_dataloader, '../models/denoising_003_model.pth', result_csv='../results/denoising_003_results.csv')
    evaluate_denoising_bilateral(denoising_003_dataloader, sigma_color=0.075, sigma_spatial=10, result_csv='../results/bilateral_denoising_003_results.csv')

    print('Evaluating deblurring, with kernel size 3')
    deblurring_3_dataloader = build_dataloader('../data/validation/blurred_3', '../data/validation/clean', data_offset=100, data_size=50)
    evaluate(deblurring_3_dataloader, '../models/deblurring_3_model.pth', result_csv='../results/deblurring_3_results.csv')
    evaluate_richardson_lucy(deblurring_3_dataloader, psf=np.ones((3, 3)) / 9, iterations=10, result_csv='../results/richardson_lucy_deblurring_3_results.csv')

    print('Evaluating deblurring, with kernel size 5')
    deblurring_5_dataloader = build_dataloader('../data/validation/blurred_5', '../data/validation/clean', data_offset=100, data_size=50)
    evaluate(deblurring_5_dataloader, '../models/deblurring_5_model.pth', result_csv='../results/deblurring_5_results.csv')
    evaluate_richardson_lucy(deblurring_5_dataloader, psf=np.ones((5, 5)) / 25, iterations=10, result_csv='../results/richardson_lucy_deblurring_5_results.csv')
