import os
import csv
import cv2
import lpips
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader

from dataset import ExposureDataset
from helpers import read_hdr, measure_ev_range, tone_map_reinhard
from model import ExposureSynthesisCNN


def tensor_to_rgb01(tensor):
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)


def rgb01_to_bgr_uint8(image_rgb01):
    image_uint8 = (np.clip(image_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)


def save_rgb01(image_path, image_rgb01):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cv2.imwrite(image_path, rgb01_to_bgr_uint8(image_rgb01))


def save_hdr_preview(image_path, hdr_rgb):
    tonemapped_image = np.clip(tone_map_reinhard(hdr_rgb), 0.0, 1.0)
    save_rgb01(image_path, tonemapped_image)


def get_tile_positions(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    positions = []
    stride = tile_size - overlap
    position = 0
    while position + tile_size < length:
        positions.append(position)
        position += stride
    positions.append(length - tile_size)
    return positions


def predict_image(model, input_tensor, device, tile_size=256, overlap=32):
    _, height, width = input_tensor.shape
    under_prediction = torch.zeros((3, height, width), device=device)
    over_prediction = torch.zeros((3, height, width), device=device)
    weights = torch.zeros((1, height, width), device=device)

    vertical_positions = get_tile_positions(height, tile_size, overlap)
    horizontal_positions = get_tile_positions(width, tile_size, overlap)

    for top in vertical_positions:
        for left in horizontal_positions:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            patch = input_tensor[:, top:bottom, left:right].unsqueeze(0).to(device)
            under_patch, over_patch = model(patch)
            under_prediction[:, top:bottom, left:right] += under_patch.squeeze(0)
            over_prediction[:, top:bottom, left:right] += over_patch.squeeze(0)
            weights[:, top:bottom, left:right] += 1.0

    return under_prediction / weights, over_prediction / weights


def build_dataloader(metadata_path, data_root='../data', data_offset=0, data_size=None, batch_size=1):
    dataset = ExposureDataset(metadata_path, data_root=data_root, data_offset=data_offset, data_size=data_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def write_exposure_results(result_csv, under_psnr_values, under_lpips_values, over_psnr_values, over_lpips_values):
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    with open(result_csv, 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['method', 'psnr', 'lpips'])
        writer.writerow(['underexposed', f'{np.mean(under_psnr_values):.4f}', f'{np.mean(under_lpips_values):.4f}'])
        writer.writerow(['overexposed', f'{np.mean(over_psnr_values):.4f}', f'{np.mean(over_lpips_values):.4f}'])


def write_hdr_results(result_csv, hdr_rows):
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    with open(result_csv, 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['image', 'dynamic_range_original', 'dynamic_range_new'])
        for row in hdr_rows:
            writer.writerow([
                row['scene_name'],
                f"{row['dynamic_range_original']:.4f}",
                f"{row['dynamic_range_new']:.4f}",
            ])


def write_sample_results(result_csv, sample_rows):
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    with open(result_csv, 'w', newline='') as result_file:
        writer = csv.DictWriter(result_file, fieldnames=['scene_name', 'target', 'psnr', 'lpips'])
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)


def merge_hdr(overexposed_image, reference_image, underexposed_image, exposure_times):
    images = [
        rgb01_to_bgr_uint8(overexposed_image),
        rgb01_to_bgr_uint8(reference_image),
        rgb01_to_bgr_uint8(underexposed_image),
    ]
    times = np.array(exposure_times, dtype=np.float32)
    merge_debevec = cv2.createMergeDebevec()
    hdr_bgr = merge_debevec.process(images, times=times)
    return cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)


def evaluate(dataloader, model_path, exposure_result_csv='../results/exposure_results.csv', hdr_result_csv='../results/hdr_results.csv'):
    print('=' * 50)
    print(f'Evaluating model: {model_path}')
    print(f'Size of test set: {len(dataloader.dataset)}')
    print(f'Saving exposure results to: {exposure_result_csv}')
    print(f'Saving HDR results to: {hdr_result_csv}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ExposureSynthesisCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    generated_root = '../results/generated'
    hdr_root = '../results/generated/hdr'
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

    under_psnr_values = []
    under_lpips_values = []
    over_psnr_values = []
    over_lpips_values = []
    sample_rows = []
    hdr_rows = []

    with torch.no_grad():
        total_batches = len(dataloader)
        for batch_index, (inputs, under_targets, over_targets, metadata) in enumerate(dataloader, start=1):
            under_targets = under_targets.to(device)
            over_targets = over_targets.to(device)
            under_outputs_list = []
            over_outputs_list = []
            for sample_index in range(inputs.shape[0]):
                under_output, over_output = predict_image(model, inputs[sample_index], device)
                under_outputs_list.append(torch.clamp(under_output, 0.0, 1.0))
                over_outputs_list.append(torch.clamp(over_output, 0.0, 1.0))

            under_outputs = torch.stack(under_outputs_list, dim=0)
            over_outputs = torch.stack(over_outputs_list, dim=0)

            under_lpips_batch = lpips_loss_fn(under_outputs * 2 - 1, under_targets * 2 - 1).cpu().numpy()
            over_lpips_batch = lpips_loss_fn(over_outputs * 2 - 1, over_targets * 2 - 1).cpu().numpy()

            for sample_index in range(inputs.shape[0]):
                scene_name = metadata['scene_name'][sample_index]

                input_image = tensor_to_rgb01(inputs[sample_index].cpu())
                under_target = tensor_to_rgb01(under_targets[sample_index].cpu())
                over_target = tensor_to_rgb01(over_targets[sample_index].cpu())
                under_output = tensor_to_rgb01(under_outputs[sample_index].cpu())
                over_output = tensor_to_rgb01(over_outputs[sample_index].cpu())

                under_psnr = psnr(under_target, under_output, data_range=1.0)
                over_psnr = psnr(over_target, over_output, data_range=1.0)
                under_lpips = float(under_lpips_batch[sample_index].item())
                over_lpips = float(over_lpips_batch[sample_index].item())

                under_psnr_values.append(under_psnr)
                under_lpips_values.append(under_lpips)
                over_psnr_values.append(over_psnr)
                over_lpips_values.append(over_lpips)

                sample_rows.append({
                    'scene_name': scene_name,
                    'target': 'underexposed',
                    'psnr': f'{under_psnr:.4f}',
                    'lpips': f'{under_lpips:.4f}',
                })
                sample_rows.append({
                    'scene_name': scene_name,
                    'target': 'overexposed',
                    'psnr': f'{over_psnr:.4f}',
                    'lpips': f'{over_lpips:.4f}',
                })

                save_rgb01(os.path.join(generated_root, 'underexposed', f'{scene_name}.png'), under_output)
                save_rgb01(os.path.join(generated_root, 'overexposed', f'{scene_name}.png'), over_output)

                hdr_image = merge_hdr(
                    over_output,
                    input_image,
                    under_output,
                    [
                        float(metadata['overexposed_exposure_time'][sample_index].item()),
                        float(metadata['reference_exposure_time'][sample_index].item()),
                        float(metadata['underexposed_exposure_time'][sample_index].item()),
                    ],
                )

                hdr_path = os.path.join(hdr_root, f'{scene_name}_generated.png')
                save_hdr_preview(hdr_path, hdr_image)

                original_hdr_path = os.path.join('../data/images/HDR', f'{scene_name}_HDR.hdr')
                original_hdr = read_hdr(original_hdr_path)
                hdr_rows.append({
                    'scene_name': scene_name,
                    'dynamic_range_original': measure_ev_range(original_hdr),
                    'dynamic_range_new': measure_ev_range(hdr_image),
                })

            if batch_index % 5 == 0 or batch_index == total_batches:
                print(f'Progress: {batch_index}/{total_batches} batches ({(100.0 * batch_index / total_batches):.1f}%)')

    write_exposure_results(
        exposure_result_csv,
        under_psnr_values,
        under_lpips_values,
        over_psnr_values,
        over_lpips_values,
    )
    write_hdr_results(hdr_result_csv, hdr_rows)
    write_sample_results('../results/exposure_samples.csv', sample_rows)

    print(f'Average underexposed PSNR: {np.mean(under_psnr_values):.4f}')
    print(f'Average underexposed LPIPS: {np.mean(under_lpips_values):.4f}')
    print(f'Average overexposed PSNR: {np.mean(over_psnr_values):.4f}')
    print(f'Average overexposed LPIPS: {np.mean(over_lpips_values):.4f}')
    print('=' * 50)

    return {
        'underexposed_psnr': float(np.mean(under_psnr_values)),
        'underexposed_lpips': float(np.mean(under_lpips_values)),
        'overexposed_psnr': float(np.mean(over_psnr_values)),
        'overexposed_lpips': float(np.mean(over_lpips_values)),
        'hdr_rows': hdr_rows,
    }


if __name__ == '__main__':
    test_dataloader = build_dataloader('../data/test/metadata.csv', batch_size=1)
    evaluate(
        test_dataloader,
        '../models/exposure_synthesis_model.pth',
        exposure_result_csv='../results/exposure_results.csv',
        hdr_result_csv='../results/hdr_results.csv',
    )