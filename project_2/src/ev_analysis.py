import csv
import os

import cv2
import numpy as np

from helpers import get_exif

TARGET_EV_OFFSET = 2.7
BRACKETED_IMAGES_FOLDER = '../data/images/Bracketed_images'
OUTPUT_SCENE_CSV = '../data/ev_pr_per_scene.csv'
OUTPUT_SUMMARY_CSV = '../data/ev_pr_summary.csv'


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


def try_read_ev(image_path):
	try:
		metadata = get_exif(image_path)
	except ValueError:
		return None

	if 'ExposureTime' not in metadata or 'FNumber' not in metadata:
		return None

	exposure_time = to_float(metadata['ExposureTime'])
	f_number = to_float(metadata['FNumber'])
	return float(np.log2((f_number ** 2) / exposure_time))


def brightness_score(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError(f'Failed to load image: {image_path}')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return float(np.percentile(gray, 60))


def nearest_ev_row(rows, target_ev, excluded_paths=None):
	if excluded_paths is None:
		excluded_paths = set()
	candidates = [row for row in rows if row['path'] not in excluded_paths]
	if not candidates:
		raise ValueError('No EV candidates available after filtering.')
	return min(candidates, key=lambda row: abs(row['ev'] - target_ev))


def index_to_percent(index, num_images):
	if num_images <= 1:
		return 0.0
	return float(index) / float(num_images - 1)


def analyze_scene(scene_name, scene_folder):
	image_paths = get_image_paths(scene_folder, image_format='.jpg')
	if not image_paths:
		return None

	brightness_rows = []
	for image_path in image_paths:
		brightness_rows.append({
			'path': image_path,
			'brightness': brightness_score(image_path),
		})

	# Index order requested: dark -> bright.
	brightness_rows.sort(key=lambda row: row['brightness'])
	path_to_brightness_index = {
		row['path']: index for index, row in enumerate(brightness_rows)
	}

	ev_rows = []
	for image_path in image_paths:
		ev_value = try_read_ev(image_path)
		if ev_value is not None:
			ev_rows.append({
				'path': image_path,
				'ev': ev_value,
			})

	if len(ev_rows) < 3:
		return {
			'scene_name': scene_name,
			'num_images': len(image_paths),
			'num_images_with_ev': len(ev_rows),
			'reference_ev': '',
			'target_under_ev': '',
			'target_over_ev': '',
			'reference_pr_dark_to_bright': '',
			'under_pr_dark_to_bright': '',
			'over_pr_dark_to_bright': '',
			'reference_image': '',
			'under_image': '',
			'over_image': '',
		}

	ev_rows.sort(key=lambda row: row['ev'])
	reference_row = ev_rows[len(ev_rows) // 2]
	target_under_ev = reference_row['ev'] + TARGET_EV_OFFSET
	target_over_ev = reference_row['ev'] - TARGET_EV_OFFSET

	under_row = nearest_ev_row(ev_rows, target_under_ev, excluded_paths={reference_row['path']})
	over_row = nearest_ev_row(ev_rows, target_over_ev, excluded_paths={reference_row['path']})

	if under_row['path'] == over_row['path']:
		over_row = nearest_ev_row(
			ev_rows,
			target_over_ev,
			excluded_paths={reference_row['path'], under_row['path']},
		)

	num_images = len(image_paths)
	reference_index = path_to_brightness_index[reference_row['path']]
	under_index = path_to_brightness_index[under_row['path']]
	over_index = path_to_brightness_index[over_row['path']]

	return {
		'scene_name': scene_name,
		'num_images': num_images,
		'num_images_with_ev': len(ev_rows),
		'reference_ev': f"{reference_row['ev']:.6f}",
		'target_under_ev': f'{target_under_ev:.6f}',
		'target_over_ev': f'{target_over_ev:.6f}',
		'reference_pr_dark_to_bright': f'{index_to_percent(reference_index, num_images):.4f}',
		'under_pr_dark_to_bright': f'{index_to_percent(under_index, num_images):.4f}',
		'over_pr_dark_to_bright': f'{index_to_percent(over_index, num_images):.4f}',
		'reference_image': os.path.basename(reference_row['path']),
		'under_image': os.path.basename(under_row['path']),
		'over_image': os.path.basename(over_row['path']),
	}


def write_csv(path, rows, field_names):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=field_names)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def main():
	scene_names = [
		name for name in sorted(os.listdir(BRACKETED_IMAGES_FOLDER))
		if os.path.isdir(os.path.join(BRACKETED_IMAGES_FOLDER, name))
	]

	scene_rows = []
	reference_percents = []
	under_percents = []
	over_percents = []

	for scene_name in scene_names:
		row = analyze_scene(scene_name, os.path.join(BRACKETED_IMAGES_FOLDER, scene_name))
		if row is None:
			continue
		scene_rows.append(row)

		if row['reference_pr_dark_to_bright'] != '':
			reference_percents.append(float(row['reference_pr_dark_to_bright']))
			under_percents.append(float(row['under_pr_dark_to_bright']))
			over_percents.append(float(row['over_pr_dark_to_bright']))

	field_names = [
		'scene_name',
		'num_images',
		'num_images_with_ev',
		'reference_ev',
		'target_under_ev',
		'target_over_ev',
		'reference_pr_dark_to_bright',
		'under_pr_dark_to_bright',
		'over_pr_dark_to_bright',
		'reference_image',
		'under_image',
		'over_image',
	]
	write_csv(OUTPUT_SCENE_CSV, scene_rows, field_names)

	ref_percent_mode = ''
	under_percent_mode = ''
	over_percent_mode = ''
	if reference_percents:
		ref_percent_mode = f'{np.median(reference_percents):.4f}'
		under_percent_mode = f'{np.median(under_percents):.4f}'
		over_percent_mode = f'{np.median(over_percents):.4f}'

	summary_rows = [
		{
			'metric': 'num_scenes_analyzed',
			'value': len(scene_rows),
		},
		{
			'metric': 'num_scenes_with_ev_triplets',
			'value': sum(1 for row in scene_rows if row['reference_pr_dark_to_bright'] != ''),
		},
		{
			'metric': 'median_reference_pr_dark_to_bright',
			'value': ref_percent_mode,
		},
		{
			'metric': 'median_under_pr_dark_to_bright',
			'value': under_percent_mode,
		},
		{
			'metric': 'median_over_pr_dark_to_bright',
			'value': over_percent_mode,
		},
		{
			'metric': 'mean_reference_pr_dark_to_bright',
			'value': f'{np.mean(reference_percents):.4f}' if reference_percents else '',
		},
		{
			'metric': 'mean_under_pr_dark_to_bright',
			'value': f'{np.mean(under_percents):.4f}' if under_percents else '',
		},
		{
			'metric': 'mean_over_pr_dark_to_bright',
			'value': f'{np.mean(over_percents):.4f}' if over_percents else '',
		},
	]
	write_csv(OUTPUT_SUMMARY_CSV, summary_rows, ['metric', 'value'])

	print(f'Saved scene-level EV index analysis to: {OUTPUT_SCENE_CSV}')
	print(f'Saved summary to: {OUTPUT_SUMMARY_CSV}')
	print(f'Median reference PR (dark->bright): {ref_percent_mode}')
	print(f'Median under PR (dark->bright): {under_percent_mode}')
	print(f'Median over PR (dark->bright): {over_percent_mode}')


if __name__ == '__main__':
	main()
