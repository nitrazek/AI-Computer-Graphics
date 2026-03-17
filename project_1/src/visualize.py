import cv2
import torch
import numpy as np
from skimage.restoration import denoise_bilateral, richardson_lucy

from model import ImageRestorationCNN


def load_image_rgb01(image_path):
    image_bgr = cv2.imread(image_path) # opencv uses BGR format, torch uses RGB
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
    return image_rgb.astype(np.float32) / 255.0

def save_image_rgb01(image_path, image_rgb01):
    image_clipped = np.clip(image_rgb01, 0.0, 1.0)
    image_uint8 = (image_clipped * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR) # convert from RGB to BGR
    cv2.imwrite(image_path, image_bgr)

def reconstruct_image(model_path, input_image_path):
    model = ImageRestorationCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # make sure to set the model to evaluation mode

    input_image = load_image_rgb01(input_image_path)
    input_tensor = torch.from_numpy(input_image.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad(): # make sure no gradient computation happens
        reconstructed_tensor = model(input_tensor)
    reconstructed_image = reconstructed_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(reconstructed_image, 0.0, 1.0)

def reconstruct_image_bilateral(input_image_path, sigma_color=0.05, sigma_spatial=15):
    input_image = load_image_rgb01(input_image_path)
    return denoise_bilateral(input_image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=-1)

def reconstruct_image_richardson_lucy(input_image_path, psf, iterations=50):
    input_image = load_image_rgb01(input_image_path)
    restored = np.stack([
        richardson_lucy(input_image[..., c], psf, num_iter=iterations)
        for c in range(input_image.shape[2])
    ], axis=2)
    return np.clip(restored, 0.0, 1.0)

if __name__ == "__main__":
    print('Visualize denoising with sigma=0.01')
    original_image = load_image_rgb01('../data/validation/clean/0807_0.png')
    blurred_image = load_image_rgb01('../data/validation/noisy_001/0807_0.png')
    reconstructed_image = reconstruct_image('../models/denoising_001_model.pth', '../data/validation/noisy_001/0807_0.png')
    reconstructed_biletral_image = reconstruct_image_bilateral('../data/validation/noisy_001/0807_0.png', sigma_color=0.025, sigma_spatial=10)
    save_image_rgb01('../visualizations/DN001_original_image.png', original_image)
    save_image_rgb01('../visualizations/DN001_noisy_image.png', blurred_image)
    save_image_rgb01('../visualizations/DN001_reconstructed_image.png', reconstructed_image)
    save_image_rgb01('../visualizations/DN001_bilateral_image.png', reconstructed_biletral_image)

    print('Visualize denoising with sigma=0.03')
    original_image = load_image_rgb01('../data/validation/clean/0802_0.png')
    blurred_image = load_image_rgb01('../data/validation/noisy_003/0802_0.png')
    reconstructed_image = reconstruct_image('../models/denoising_003_model.pth', '../data/validation/noisy_003/0802_0.png')
    reconstructed_image_bilateral = reconstruct_image_bilateral('../data/validation/noisy_003/0802_0.png', sigma_color=0.075, sigma_spatial=10)
    save_image_rgb01('../visualizations/DN003_original_image.png', original_image)
    save_image_rgb01('../visualizations/DN003_noisy_image.png', blurred_image)
    save_image_rgb01('../visualizations/DN003_reconstructed_image.png', reconstructed_image)
    save_image_rgb01('../visualizations/DN003_bilateral_image.png', reconstructed_image_bilateral)

    print('Visualize deblurring with kernel size 3')
    original_image = load_image_rgb01('../data/validation/clean/0803_0.png')
    blurred_image = load_image_rgb01('../data/validation/blurred_3/0803_0.png')
    reconstructed_image = reconstruct_image('../models/deblurring_3_model.pth', '../data/validation/blurred_3/0803_0.png')
    reconstructed_rl_image = reconstruct_image_richardson_lucy('../data/validation/blurred_3/0803_0.png', psf=np.ones((3, 3)) / 9, iterations=10)
    save_image_rgb01('../visualizations/DB003_original_image.png', original_image)
    save_image_rgb01('../visualizations/DB003_blurred_image.png', blurred_image)
    save_image_rgb01('../visualizations/DB003_reconstructed_image.png', reconstructed_image)
    save_image_rgb01('../visualizations/DB003_richardson_lucy_image.png', reconstructed_rl_image)

    print('Visualize deblurring with kernel size 5')
    original_image = load_image_rgb01('../data/validation/clean/0804_0.png')
    blurred_image = load_image_rgb01('../data/validation/blurred_5/0804_0.png')
    reconstructed_image = reconstruct_image('../models/deblurring_5_model.pth', '../data/validation/blurred_5/0804_0.png')
    reconstructed_rl_image = reconstruct_image_richardson_lucy('../data/validation/blurred_5/0804_0.png', psf=np.ones((5, 5)) / 25, iterations=10)
    save_image_rgb01('../visualizations/DB005_original_image.png', original_image)
    save_image_rgb01('../visualizations/DB005_blurred_image.png', blurred_image)
    save_image_rgb01('../visualizations/DB005_reconstructed_image.png', reconstructed_image)
    save_image_rgb01('../visualizations/DB005_richardson_lucy_image.png', reconstructed_rl_image)
