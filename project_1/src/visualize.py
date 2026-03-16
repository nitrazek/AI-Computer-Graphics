import cv2
import torch

from model import ImageRestorationCNN

def reconstruct_image(model_path, input_image_path):
    model = ImageRestorationCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    input_image = cv2.imread(input_image_path)
    input_tensor = torch.from_numpy(input_image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0 # type: ignore
    reconstructed_tensor = model(input_tensor).squeeze(0).detach().numpy().transpose(1, 2, 0) * 255.0
    return reconstructed_tensor.astype('uint8')

if __name__ == "__main__":
    print('Visualize denoising with sigma=0.01')
    original_image = cv2.imread('../data/validation/clean/0801_0.png')
    blurred_image = cv2.imread('../data/validation/noisy_001/0801_0.png')
    reconstructed_image = reconstruct_image('../models/denoising_001_model.pth', '../data/validation/noisy_001/0801_0.png')

    cv2.imwrite('../visualizations/DN001_original_image.png', original_image) # type: ignore
    cv2.imwrite('../visualizations/DN001_noisy_image.png', blurred_image) # type: ignore
    cv2.imwrite('../visualizations/DN001_reconstructed_image.png', reconstructed_image)

    print('Visualize denoising with sigma=0.03')
    original_image = cv2.imread('../data/validation/clean/0802_0.png')
    blurred_image = cv2.imread('../data/validation/noisy_003/0802_0.png')
    reconstructed_image = reconstruct_image('../models/denoising_003_model.pth', '../data/validation/noisy_003/0802_0.png')

    cv2.imwrite('../visualizations/DN003_original_image.png', original_image) # type: ignore
    cv2.imwrite('../visualizations/DN003_noisy_image.png', blurred_image) # type: ignore
    cv2.imwrite('../visualizations/DN003_reconstructed_image.png', reconstructed_image)

    print('Visualize deblurring with kernel size 3')
    original_image = cv2.imread('../data/validation/clean/0803_0.png')
    blurred_image = cv2.imread('../data/validation/blurred_3/0803_0.png')
    reconstructed_image = reconstruct_image('../models/deblurring_3_model.pth', '../data/validation/blurred_3/0803_0.png')
    cv2.imwrite('../visualizations/DB003_original_image.png', original_image) # type: ignore
    cv2.imwrite('../visualizations/DB003_blurred_image.png', blurred_image) # type: ignore
    cv2.imwrite('../visualizations/DB003_reconstructed_image.png', reconstructed_image)

    print('Visualize deblurring with kernel size 5')
    original_image = cv2.imread('../data/validation/clean/0804_0.png')
    blurred_image = cv2.imread('../data/validation/blurred_5/0804_0.png')
    reconstructed_image = reconstruct_image('../models/deblurring_5_model.pth', '../data/validation/blurred_5/0804_0.png')
    cv2.imwrite('../visualizations/DB005_original_image.png', original_image) # type: ignore
    cv2.imwrite('../visualizations/DB005_blurred_image.png', blurred_image) # type: ignore
    cv2.imwrite('../visualizations/DB005_reconstructed_image.png', reconstructed_image)
