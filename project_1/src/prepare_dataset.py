import os
import cv2
import numpy as np
from skimage.util import random_noise

def get_image_paths(folder):
    image_paths = []
    for file in os.listdir(folder):
        if file.endswith(('.png')):
            image_paths.append(os.path.join(folder, file))

    return image_paths

def random_crop(image, crop_size=256):
    if image.shape[0] < crop_size or image.shape[1] < crop_size:
        raise ValueError('Crop size must be smaller than image dimensions.')

    x = np.random.randint(0, image.shape[0] - crop_size + 1)
    y = np.random.randint(0, image.shape[1] - crop_size + 1)
    return image[x : x + crop_size, y : y + crop_size]

def add_gaussian_noise(image, sigma):
    # pixel range must be in [0, 1], convert to float and normalize
    image = image.astype(np.float32) / 255.0
    # add Gaussian noise with given sigma
    noisy_image = random_noise(image, mode='gaussian', var=sigma**2)
    # convert back to uint8
    noisy_image = (noisy_image * 255).astype(np.uint8)

    return noisy_image

def add_gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


if __name__ == "__main__":

    n_patch = 4
    training_image_paths = get_image_paths('../data/DIV2K_train_HR')
    os.makedirs('../data/training/clean', exist_ok=True)
    os.makedirs('../data/training/noisy_001', exist_ok=True)
    os.makedirs('../data/training/noisy_003', exist_ok=True)
    os.makedirs('../data/training/blurred_3', exist_ok=True)
    os.makedirs('../data/training/blurred_5', exist_ok=True)

    for path in training_image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f'Failed to read image: {path}')
            continue

        for i in range(n_patch):
            random_croped_image = random_crop(image)
            noisy_image_001 = add_gaussian_noise(random_croped_image, sigma=0.01)
            noisy_image_003 = add_gaussian_noise(random_croped_image, sigma=0.03)
            blurred_image_3 = add_gaussian_blur(random_croped_image, kernel_size=3)
            blurred_image_5 = add_gaussian_blur(random_croped_image, kernel_size=5)

            cv2.imwrite('../data/training/clean/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), random_croped_image)
            cv2.imwrite('../data/training/noisy_001/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), noisy_image_001)
            cv2.imwrite('../data/training/noisy_003/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), noisy_image_003)
            cv2.imwrite('../data/training/blurred_3/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), blurred_image_3)
            cv2.imwrite('../data/training/blurred_5/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), blurred_image_5)

    validation_image_paths = get_image_paths('../data/DIV2K_valid_HR')
    os.makedirs('../data/validation/clean', exist_ok=True)
    os.makedirs('../data/validation/noisy_001', exist_ok=True)
    os.makedirs('../data/validation/noisy_003', exist_ok=True)
    os.makedirs('../data/validation/blurred_3', exist_ok=True)
    os.makedirs('../data/validation/blurred_5', exist_ok=True)

    for path in validation_image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f'Failed to read image: {path}')
            continue

        for i in range(n_patch):
            random_croped_image = random_crop(image)
            noisy_image_001 = add_gaussian_noise(random_croped_image, sigma=0.01)
            noisy_image_003 = add_gaussian_noise(random_croped_image, sigma=0.03)
            blurred_image_3 = add_gaussian_blur(random_croped_image, kernel_size=3)
            blurred_image_5 = add_gaussian_blur(random_croped_image, kernel_size=5)

            cv2.imwrite('../data/validation/clean/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), random_croped_image)
            cv2.imwrite('../data/validation/noisy_001/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), noisy_image_001)
            cv2.imwrite('../data/validation/noisy_003/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), noisy_image_003)
            cv2.imwrite('../data/validation/blurred_3/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), blurred_image_3)
            cv2.imwrite('../data/validation/blurred_5/{}_{}.png'.format(os.path.splitext(os.path.basename(path))[0], i), blurred_image_5)