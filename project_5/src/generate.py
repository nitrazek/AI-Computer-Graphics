"""Generate stickman animations conditioned by text prompt ('walk' or 'jump')."""
import argparse
import os
import numpy as np

from inference import load_model_and_diffusion, label_from_prompt, sample_motion
from visualize import animate_skeleton_3d, render_pose_strip


GENERATED_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'generated')


def generate(prompt, n_samples=3, model_path=None):
    label_name, label_index = label_from_prompt(prompt)
    model, diffusion, device = load_model_and_diffusion(model_path=model_path)

    motions = sample_motion(
        model,
        diffusion,
        label_index=label_index,
        n_samples=n_samples,
        device=device,
    )

    os.makedirs(GENERATED_DIR, exist_ok=True)
    npy_path = os.path.join(GENERATED_DIR, f'{label_name}_samples.npy')
    np.save(npy_path, motions)

    for index in range(n_samples):
        gif_path = os.path.join(GENERATED_DIR, f'{label_name}_{index:02d}.gif')
        png_path = os.path.join(GENERATED_DIR, f'{label_name}_{index:02d}.png')
        animate_skeleton_3d(motions[index], output_filename=gif_path, fps=24)
        render_pose_strip(motions[index], output_path=png_path, title=f'Generated {label_name}')

    print(f'Prompt: {prompt}')
    print(f'Generated label: {label_name}')
    print(f'Saved samples tensor: {npy_path}')
    print(f'Saved {n_samples} gifs and pose strips to: {GENERATED_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate stickman motion from text prompt.')
    parser.add_argument('--prompt', type=str, required=True, help="Text containing 'walk' or 'jump'.")
    parser.add_argument('--n-samples', type=int, default=3, help='Number of samples to generate.')
    parser.add_argument('--model-path', type=str, default=None, help='Path to trained model checkpoint.')
    args = parser.parse_args()

    generate(prompt=args.prompt, n_samples=args.n_samples, model_path=args.model_path)
