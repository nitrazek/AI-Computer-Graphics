import os
import cv2
import torch
import numpy as np

from dataset import ExposureDataset
from helpers import read_hdr, tone_map_reinhard
from model import ExposureSynthesisCNN
from evaluate import merge_hdr, predict_image


def save_image_rgb01(image_path, image_rgb01):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image_uint8 = (np.clip(image_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr)


def tensor_to_rgb01(tensor):
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)


def create_comparison_row(images):
    clipped_images = [np.clip(image, 0.0, 1.0) for image in images]
    return np.concatenate(clipped_images, axis=1)


def visualize(model_path, metadata_path='../data/test/metadata.csv', data_root='../data', output_dir='../visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    dataset = ExposureDataset(metadata_path, data_root=data_root)

    model = ExposureSynthesisCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        for index in range(len(dataset)):
            input_image, under_target, over_target, metadata = dataset[index]

            input_rgb = tensor_to_rgb01(input_image)
            under_target_rgb = tensor_to_rgb01(under_target)
            over_target_rgb = tensor_to_rgb01(over_target)
            under_output, over_output = predict_image(model, input_image, torch.device('cpu'))
            under_output_rgb = np.clip(tensor_to_rgb01(under_output), 0.0, 1.0)
            over_output_rgb = np.clip(tensor_to_rgb01(over_output), 0.0, 1.0)

            hdr_generated = merge_hdr(
                over_output_rgb,
                input_rgb,
                under_output_rgb,
                [
                    float(metadata['overexposed_exposure_time'].item()),
                    float(metadata['reference_exposure_time'].item()),
                    float(metadata['underexposed_exposure_time'].item()),
                ],
            )
            hdr_original = read_hdr(os.path.join(data_root, 'images', 'HDR', f"{metadata['scene_name']}_HDR.hdr"))

            hdr_generated_tonemapped = np.clip(tone_map_reinhard(hdr_generated), 0.0, 1.0)
            hdr_original_tonemapped = np.clip(tone_map_reinhard(hdr_original), 0.0, 1.0)

            exposure_comparison = create_comparison_row([
                input_rgb,
                under_target_rgb,
                under_output_rgb,
                over_target_rgb,
                over_output_rgb,
            ])
            hdr_comparison = create_comparison_row([
                hdr_original_tonemapped,
                hdr_generated_tonemapped,
            ])

            scene_name = metadata['scene_name']
            save_image_rgb01(os.path.join(output_dir, f'{scene_name}_exposure_comparison.png'), exposure_comparison)
            save_image_rgb01(os.path.join(output_dir, f'{scene_name}_hdr_comparison.png'), hdr_comparison)


if __name__ == '__main__':
    visualize('../models/exposure_synthesis_model.pth')