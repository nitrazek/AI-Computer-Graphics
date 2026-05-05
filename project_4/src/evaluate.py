import os
import numpy as np
import torch

from helpers import (
    load_obj,
    normalize_mesh,
    sample_points_from_mesh,
    random_rotation_matrix,
    chamfer_distance,
    iou_and_dice,
)
from model import VectorFieldNet


def write_results(result_csv, rows):
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)

    with open(result_csv, 'w') as f:
        f.write('method,iou,dice,chamfer\n')
        for method, iou, dice, chamfer in rows:
            f.write(f"{method},{iou:.6f},{dice:.6f},{chamfer:.6f}\n")


def evaluate_flow(
    model_path,
    source_obj_path,
    target_obj_path,
    n_source_points=4096,
    n_target_points=4096,
    n_integration_steps=8,
    n_trials=5,
    voxel_resolution=64,
    augment=True,
    seed=0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_vertices, source_faces = load_obj(source_obj_path)
    target_vertices, target_faces = load_obj(target_obj_path)
    source_vertices = normalize_mesh(source_vertices)
    target_vertices = normalize_mesh(target_vertices)

    model = VectorFieldNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    iou_values, dice_values, chamfer_values = [], [], []
    rng = np.random.default_rng(seed)

    target_points_eval = sample_points_from_mesh(
        target_vertices, target_faces, n_target_points, rng=rng
    )

    with torch.no_grad():
        for trial in range(n_trials):
            source_points = sample_points_from_mesh(
                source_vertices, source_faces, n_source_points, rng=rng
            )

            if augment:
                rotation = random_rotation_matrix(rng=rng)
                scale = float(rng.uniform(0.7, 1.3))
                source_points = (source_points @ rotation.T) * scale

            source_tensor = torch.from_numpy(source_points).unsqueeze(0).to(device)
            predicted = model(source_tensor, n_steps=n_integration_steps)
            predicted_np = predicted.squeeze(0).cpu().numpy()

            chamfer_values.append(chamfer_distance(predicted_np, target_points_eval))
            iou, dice = iou_and_dice(predicted_np, target_points_eval, resolution=voxel_resolution)
            iou_values.append(iou)
            dice_values.append(dice)

    return (
        float(np.mean(iou_values)),
        float(np.mean(dice_values)),
        float(np.mean(chamfer_values)),
    )


if __name__ == '__main__':
    source_objects = [
        ('bunny', '../models/bunny.obj'),
        ('dragon', '../models/dragon_small.obj'),
        ('armadillo', '../models/armadillo_small.obj'),
    ]
    asian_dragon_path = '../models/asian_dragon_really_small.obj'
    teapot_path = '../models/teapot.obj'

    rows = []

    print('=' * 50)
    print('Evaluating each flow on its native source object')
    for name, source_path in source_objects:
        model_path = f'../models/{name}_flow.pth'
        iou, dice, chamfer = evaluate_flow(
            model_path=model_path,
            source_obj_path=source_path,
            target_obj_path=teapot_path,
        )
        method = f'{name} flow'
        rows.append((method, iou, dice, chamfer))
        print(f'{method:>32} | IoU: {iou:.4f} | Dice: {dice:.4f} | Chamfer: {chamfer:.6f}')

    print('=' * 50)
    print('Evaluating each flow on the Asian Dragon (out-of-distribution input)')
    for name, _ in source_objects:
        model_path = f'../models/{name}_flow.pth'
        iou, dice, chamfer = evaluate_flow(
            model_path=model_path,
            source_obj_path=asian_dragon_path,
            target_obj_path=teapot_path,
        )
        method = f'{name} flow asian dragon'
        rows.append((method, iou, dice, chamfer))
        print(f'{method:>32} | IoU: {iou:.4f} | Dice: {dice:.4f} | Chamfer: {chamfer:.6f}')

    write_results('../results/transformation_results.csv', rows)
    print('=' * 50)
    print('Results saved to ../results/transformation_results.csv')
