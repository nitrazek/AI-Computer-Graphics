import os
import cv2
import numpy as np
from helpers import get_exif

def get_image_paths(folder, format='.png'):
    image_paths = []
    for file in os.listdir(folder):
        if file.endswith((format.lower(), format.upper())):
            image_paths.append(os.path.join(folder, file))

    return image_paths

def get_exposure_time(image_path):
    metadata = get_exif(image_path)
    return metadata.get('ExposureTime')

def get_exposure_value(image_path):
    metadata = get_exif(image_path)
    return np.log2(float(metadata.get('FNumber')) ** 2 / float(metadata.get('ExposureTime')))

if __name__ == "__main__":
    LDR_folder = "../data/images/LDR/"
    Bracketed_images_folder = "../data/images/Bracketed_images/"
    image_names = [f'C{'0' if i < 10 else ''}{i}' for i in range(1, 40)]

    for image_name in image_names:
        ldr_image_path = LDR_folder + image_name + '_LDR.tif'
        bracketed_image_paths = get_image_paths(Bracketed_images_folder + image_name, format='.JPG')

        bracketed_image_paths.sort(key=lambda x: get_exposure_value(x))
        l = len(bracketed_image_paths)
        min_ev = float(get_exposure_value(bracketed_image_paths[0]))
        max_ev = float(get_exposure_value(bracketed_image_paths[-1]))
        print(min_ev, max_ev)
