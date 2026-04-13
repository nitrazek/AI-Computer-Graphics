# For images without metadata, we will estimate the EV based on the image brightness and sort the images by brightness to select the reference, underexposed, and overexposed images.
# Reference image = image at PR=0.5
# Underexposed image = image at PR=0
# Overexposed image = image at PR=0.875
# PR values above is calculated by ev_analysis.py.

import csv
import os
import cv2
import numpy as np
from helpers import get_exif

TARGET_EV_OFFSET = 2.7
VALIDATION_SCENES = ['C35', 'C36', 'C37', 'C38']
TEST_SCENES = [f'C{i:02d}' for i in range(40, 47)]

LDR_FOLDER = '../data/images/LDR'
BRACKETED_IMAGES_FOLDER = '../data/images/Bracketed_images'
OUTPUT_FOLDER = '../data'

REFERENCE_PR = 0.5
UNDEREXPOSED_PR = 0.0
OVEREXPOSED_PR = 0.875


def get_scene_names(folder):
    scene_names = []
    for file_name in os.listdir(folder):
        if file_name.endswith('_LDR.tif'):
            scene_names.append(file_name.replace('_LDR.tif', ''))

    scene_names.sort()
    return scene_names


def get_image_paths(folder, image_format='.jpg'):
    image_paths = []
    for file_name in os.listdir(folder):
        if file_name.endswith((image_format.lower(), image_format.upper())):
            image_paths.append(os.path.join(folder, file_name))

    image_paths.sort()
    return image_paths


def to_float(value):
    try:
        return float(value)
    except TypeError:
        return float(value[0]) / float(value[1])


def get_bracket_metadata(image_path):
    metadata = get_exif(image_path)
    if 'ExposureTime' not in metadata or 'FNumber' not in metadata:
        raise ValueError(f'Missing exposure metadata for image: {image_path}')

    exposure_time = to_float(metadata.get('ExposureTime'))
    f_number = to_float(metadata.get('FNumber'))
    exposure_value = np.log2(f_number ** 2 / exposure_time)

    return {
        'path': image_path,
        'exposure_time': exposure_time,
        'f_number': f_number,
        'ev': float(exposure_value),
    }


def select_nearest_image(image_data, target_ev, excluded_paths=None):
    if excluded_paths is None:
        excluded_paths = set()

    filtered_data = [image for image in image_data if image['path'] not in excluded_paths]
    if not filtered_data:
        raise ValueError('No bracketed images available after filtering.')

    return min(filtered_data, key=lambda image: abs(image['ev'] - target_ev))


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Failed to load image: {image_path}')

    return image


def brightness_score(image_path):
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.percentile(gray, 60))


def pr_to_index(pr_value, num_images):
    if num_images <= 1:
        return 0
    return int(round(pr_value * (num_images - 1)))


def fallback_select_by_pr(bracketed_image_paths):
    if len(bracketed_image_paths) < 3:
        raise ValueError('Not enough bracketed images to select fallback targets.')

    brightness_rows = []
    for path in bracketed_image_paths:
        brightness_rows.append({
            'path': path,
            'brightness': brightness_score(path),
        })

    # Dark -> bright ordering, consistent with PR analysis.
    brightness_rows.sort(key=lambda row: row['brightness'])
    num_images = len(brightness_rows)

    reference_index = pr_to_index(REFERENCE_PR, num_images)
    under_index = pr_to_index(UNDEREXPOSED_PR, num_images)
    over_index = pr_to_index(OVEREXPOSED_PR, num_images)

    chosen = {reference_index}
    if under_index in chosen:
        for candidate in range(num_images):
            if candidate not in chosen:
                under_index = candidate
                break
    chosen.add(under_index)
    if over_index in chosen:
        for candidate in range(num_images - 1, -1, -1):
            if candidate not in chosen:
                over_index = candidate
                break

    reference_path = brightness_rows[reference_index]['path']
    under_path = brightness_rows[under_index]['path']
    over_path = brightness_rows[over_index]['path']

    reference_exposure_time = 1.0 / 60.0
    under_exposure_time = reference_exposure_time * (2.0 ** (-TARGET_EV_OFFSET))
    over_exposure_time = reference_exposure_time * (2.0 ** TARGET_EV_OFFSET)

    reference_image = {
        'path': reference_path,
        'exposure_time': float(reference_exposure_time),
        'ev': 0.0,
    }
    underexposed_image = {
        'path': under_path,
        'exposure_time': float(under_exposure_time),
        'ev': TARGET_EV_OFFSET,
    }
    overexposed_image = {
        'path': over_path,
        'exposure_time': float(over_exposure_time),
        'ev': -TARGET_EV_OFFSET,
    }
    return reference_image, underexposed_image, overexposed_image


def resize_image(image, target_shape):
    target_height, target_width = target_shape[:2]
    if image.shape[0] == target_height and image.shape[1] == target_width:
        return image

    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def save_image(image_path, image):
    if not cv2.imwrite(image_path, image):
        raise ValueError(f'Failed to save image: {image_path}')


def get_split_name(scene_name):
    if scene_name in TEST_SCENES:
        return 'test'
    if scene_name in VALIDATION_SCENES:
        return 'validation'

    return 'training'


def create_output_dirs():
    output_dirs = {}
    for split_name in ['training', 'validation', 'test']:
        output_dirs[split_name] = {
            'ldr': os.path.join(OUTPUT_FOLDER, split_name, 'ldr'),
            'underexposed': os.path.join(OUTPUT_FOLDER, split_name, 'underexposed'),
            'overexposed': os.path.join(OUTPUT_FOLDER, split_name, 'overexposed'),
        }

        for folder in output_dirs[split_name].values():
            os.makedirs(folder, exist_ok=True)

    return output_dirs


def prepare_scene(scene_name, output_dirs):
    ldr_image_path = os.path.join(LDR_FOLDER, f'{scene_name}_LDR.tif')
    bracketed_folder = os.path.join(BRACKETED_IMAGES_FOLDER, scene_name)

    if not os.path.exists(ldr_image_path):
        raise ValueError(f'Missing LDR image: {ldr_image_path}')
    if not os.path.isdir(bracketed_folder):
        raise ValueError(f'Missing bracketed images folder: {bracketed_folder}')

    bracketed_image_paths = get_image_paths(bracketed_folder, image_format='.jpg')
    bracketed_image_data = []
    for image_path in bracketed_image_paths:
        try:
            bracketed_image_data.append(get_bracket_metadata(image_path))
        except ValueError:
            continue

    if len(bracketed_image_data) >= 3:
        bracketed_image_data.sort(key=lambda image: image['ev'])

        reference_image = bracketed_image_data[len(bracketed_image_data) // 2]
        under_target_ev = reference_image['ev'] + TARGET_EV_OFFSET
        over_target_ev = reference_image['ev'] - TARGET_EV_OFFSET

        underexposed_image = select_nearest_image(
            bracketed_image_data,
            under_target_ev,
            excluded_paths={reference_image['path']}
        )
        overexposed_image = select_nearest_image(
            bracketed_image_data,
            over_target_ev,
            excluded_paths={reference_image['path']}
        )

        if overexposed_image['path'] == underexposed_image['path']:
            overexposed_image = select_nearest_image(
                bracketed_image_data,
                over_target_ev,
                excluded_paths={reference_image['path'], underexposed_image['path']}
            )
    else:
        reference_image, underexposed_image, overexposed_image = fallback_select_by_pr(bracketed_image_paths)
        under_target_ev = TARGET_EV_OFFSET
        over_target_ev = -TARGET_EV_OFFSET

    ldr_image = load_image(ldr_image_path)
    underexposed_target = resize_image(load_image(underexposed_image['path']), ldr_image.shape)
    overexposed_target = resize_image(load_image(overexposed_image['path']), ldr_image.shape)

    split_name = get_split_name(scene_name)
    ldr_output_path = os.path.join(output_dirs[split_name]['ldr'], f'{scene_name}.png')
    under_output_path = os.path.join(output_dirs[split_name]['underexposed'], f'{scene_name}.png')
    over_output_path = os.path.join(output_dirs[split_name]['overexposed'], f'{scene_name}.png')

    save_image(ldr_output_path, ldr_image)
    save_image(under_output_path, underexposed_target)
    save_image(over_output_path, overexposed_target)

    return {
        'scene_name': scene_name,
        'split': split_name,
        'ldr_path': os.path.relpath(ldr_output_path, OUTPUT_FOLDER),
        'underexposed_path': os.path.relpath(under_output_path, OUTPUT_FOLDER),
        'overexposed_path': os.path.relpath(over_output_path, OUTPUT_FOLDER),
        'reference_image_path': reference_image['path'],
        'underexposed_image_path': underexposed_image['path'],
        'overexposed_image_path': overexposed_image['path'],
        'reference_exposure_time': reference_image['exposure_time'],
        'underexposed_exposure_time': underexposed_image['exposure_time'],
        'overexposed_exposure_time': overexposed_image['exposure_time'],
        'reference_ev': reference_image['ev'],
        'underexposed_ev': underexposed_image['ev'],
        'overexposed_ev': overexposed_image['ev'],
        'underexposed_ev_error': abs(underexposed_image['ev'] - under_target_ev),
        'overexposed_ev_error': abs(overexposed_image['ev'] - over_target_ev),
    }


def write_metadata(metadata_rows, output_dirs):
    field_names = [
        'scene_name',
        'split',
        'ldr_path',
        'underexposed_path',
        'overexposed_path',
        'reference_image_path',
        'underexposed_image_path',
        'overexposed_image_path',
        'reference_exposure_time',
        'underexposed_exposure_time',
        'overexposed_exposure_time',
        'reference_ev',
        'underexposed_ev',
        'overexposed_ev',
        'underexposed_ev_error',
        'overexposed_ev_error',
    ]

    for split_name in ['training', 'validation', 'test']:
        metadata_path = os.path.join(OUTPUT_FOLDER, split_name, 'metadata.csv')
        with open(metadata_path, 'w', newline='') as metadata_file:
            writer = csv.DictWriter(metadata_file, fieldnames=field_names)
            writer.writeheader()
            for row in metadata_rows[split_name]:
                writer.writerow(row)


def write_skipped_scenes(skipped_rows):
    skipped_scenes_path = os.path.join(OUTPUT_FOLDER, 'skipped_scenes.csv')
    with open(skipped_scenes_path, 'w', newline='') as skipped_file:
        writer = csv.DictWriter(skipped_file, fieldnames=['scene_name', 'reason'])
        writer.writeheader()
        for row in skipped_rows:
            writer.writerow(row)


if __name__ == "__main__":
    output_dirs = create_output_dirs()
    scene_names = get_scene_names(LDR_FOLDER)

    metadata_rows = {
        'training': [],
        'validation': [],
        'test': [],
    }
    skipped_rows = []

    for scene_name in scene_names:
        try:
            metadata = prepare_scene(scene_name, output_dirs)
            metadata_rows[metadata['split']].append(metadata)
            print(f'Prepared {scene_name} ({metadata["split"]})')
        except ValueError as error:
            skipped_rows.append({
                'scene_name': scene_name,
                'reason': str(error),
            })
            print(f'Skipped {scene_name}: {error}')

    write_metadata(metadata_rows, output_dirs)
    write_skipped_scenes(skipped_rows)

    print(f'Training scenes: {len(metadata_rows["training"])}')
    print(f'Validation scenes: {len(metadata_rows["validation"])}')
    print(f'Test scenes: {len(metadata_rows["test"])}')
    print(f'Skipped scenes: {len(skipped_rows)}')
