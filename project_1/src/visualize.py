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
    original_image = cv2.imread('../data/validation/clean/0900_0.png')
    blurred_image = cv2.imread('../data/validation/blurred_3/0900_0.png')

    model = ImageRestorationCNN()
    model.load_state_dict(torch.load('../models/denoising_model.pth', map_location=torch.device('cpu')))

    reconstructed_image = reconstruct_image('../models/denoising_model.pth', '../data/validation/blurred_3/0900_0.png')

    cv2.imwrite('original_image.png', original_image) # type: ignore
    cv2.imwrite('blurred_image.png', blurred_image) # type: ignore
    cv2.imwrite('reconstructed_image.png', reconstructed_image)

