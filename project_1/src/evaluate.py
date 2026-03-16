import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_bilateral, richardson_lucy
import lpips

from dataset import ImageDataset
from model import ImageRestorationCNN

def evaluate(dataloader, model_path, result_csv='result.csv'):
    print('=' * 50)
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
    print('=' * 50)

    with open(result_csv, 'w') as f:
        f.write('PSNR, SSIM, LPIPS\n')
        f.write(f"Average PSNR: {np.mean(psnr_values):.4f}\n")
        f.write(f"Average SSIM: {np.mean(ssim_values):.4f}\n")
        f.write(f"Average LPIPS: {np.mean(lpips_values):.4f}\n")
        for psnr_val, ssim_val, lpips_val in zip(psnr_values, ssim_values, lpips_values):
            f.write(f"{psnr_val:.4f}, {ssim_val:.4f}, {lpips_val:.4f}\n")

def evaluate_external_method(dataloader, restoration_method, result_csv='external_method_results.csv'):
    print('=' * 50)
    print(f"Evaluating external method: {restoration_method}")
    print(f"Size of test set: {len(dataloader.dataset)}")
    print(f"Saving results to: {result_csv}")
    
    psnr_values, ssim_values, lpips_values = [], [], []
    lpips_loss_fn = lpips.LPIPS(net='alex')

    # Small normalized blur kernel used for Richardson-Lucy deblurring.
    psf = np.ones((5, 5), dtype=np.float32)
    psf /= psf.sum()

    for inputs, targets in dataloader:
        inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        targets_np = targets.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        for i in range(inputs_np.shape[0]):
            restored = None
            if restoration_method == 'bilateral_denoising':
                restored = denoise_bilateral(inputs_np[i], channel_axis=-1)
            elif restoration_method == 'richardson_lucy_deblurring':
                # richardson_lucy expects image and PSF dimensionality to match.
                # For RGB images, apply RL independently per channel with a 2D PSF.
                restored = np.stack(
                    [
                        richardson_lucy(inputs_np[i, :, :, c], psf, num_iter=10)
                        for c in range(inputs_np[i].shape[2])
                    ],
                    axis=-1,
                )

            restored = np.clip(restored, 0.0, 1.0)
            
            psnr_values.append(psnr(targets_np[i], restored, data_range=1.0))
            ssim_values.append(ssim(targets_np[i], restored, channel_axis=2, data_range=1.0))
            
            restored_lpips = torch.from_numpy(restored.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1  # [0, 1] -> [-1, 1]
            targets_lpips = torch.from_numpy(targets_np[i].transpose(2, 0, 1)).unsqueeze(0) * 2 - 1  # [0, 1] -> [-1, 1]
            lpips_value = lpips_loss_fn(restored_lpips.float(), targets_lpips.float()).item()
            lpips_values.append(lpips_value)

    print(f"Average PSNR: {np.mean(psnr_values):.4f}")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_values):.4f}")
    print('=' * 50)

    with open(result_csv, 'w') as f:
        f.write('PSNR, SSIM, LPIPS\n')
        f.write(f"Average PSNR: {np.mean(psnr_values):.4f}\n")
        f.write(f"Average SSIM: {np.mean(ssim_values):.4f}\n")
        f.write(f"Average LPIPS: {np.mean(lpips_values):.4f}\n")
        for psnr_val, ssim_val, lpips_val in zip(psnr_values, ssim_values, lpips_values):
            f.write(f"{psnr_val:.4f}, {ssim_val:.4f}, {lpips_val:.4f}\n")


if __name__ == "__main__":
    dataset = ImageDataset(input_paths='../data/validation/noisy_001', target_paths='../data/validation/clean', data_offset=0, data_size=50)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    evaluate(dataloader, '../models/denoising_model.pth', result_csv='../results/denoising_results.csv')
    evaluate(dataloader, '../models/deblurring_model.pth', result_csv='../results/deblurring_results.csv')
    evaluate_external_method(dataloader, 'bilateral_denoising', result_csv='../results/bilateral_denoising_results.csv')
    evaluate_external_method(dataloader, 'richardson_lucy_deblurring', result_csv='../results/richardson_lucy_deblurring_results.csv')