"""Evaluate generated motions with FMD, MPJPE, and sample variance for walk/jump."""
import argparse
import os
import numpy as np
from scipy.linalg import sqrtm

from inference import load_model_and_diffusion, sample_motion
from skeleton import LABEL_NAMES, LABEL_TO_INDEX


PROCESSED_VALIDATION_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'validation'
)
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'evaluation_results.csv')


def flatten_motion(samples):
    # samples: [N, 48, 15, 3] -> [N, 2160]
    return samples.reshape(samples.shape[0], -1)


def frechet_distance(real, generated):
    real_flat = flatten_motion(real)
    gen_flat = flatten_motion(generated)

    mu_real = np.mean(real_flat, axis=0)
    mu_gen = np.mean(gen_flat, axis=0)
    sigma_real = np.cov(real_flat, rowvar=False)
    sigma_gen = np.cov(gen_flat, rowvar=False)

    mean_diff = mu_real - mu_gen
    cov_prod = sigma_real @ sigma_gen
    cov_sqrt = sqrtm(cov_prod)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fmd = mean_diff @ mean_diff + np.trace(sigma_real + sigma_gen - 2.0 * cov_sqrt)
    return float(max(fmd, 0.0))


def mpjpe_nearest(real, generated):
    """Nearest-neighbor MPJPE between generated samples and validation samples."""
    # pairwise distance over [T, J, C]
    diff = generated[:, None, :, :, :] - real[None, :, :, :, :]
    # average over T, J, C gives per-pair MSE^0.5
    pair_errors = np.sqrt(np.mean(np.square(diff), axis=(2, 3, 4)))
    nearest_errors = np.min(pair_errors, axis=1)
    return float(np.mean(nearest_errors))


def motion_variance(samples):
    # Average variance across all coordinates over generated sample axis.
    return float(np.mean(np.var(samples, axis=0)))


def write_results(rows, output_path=RESULTS_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('motion,fmd,mpjpe,var\n')
        for motion_name, fmd, mpjpe, variance in rows:
            handle.write(f'{motion_name},{fmd:.6f},{mpjpe:.6f},{variance:.6f}\n')


def evaluate(model_path=None, n_samples_per_label=128):
    model, diffusion, device = load_model_and_diffusion(model_path=model_path)
    rows = []

    for motion_name in LABEL_NAMES:
        real_path = os.path.join(PROCESSED_VALIDATION_DIR, f'{motion_name}.npy')
        real = np.load(real_path).astype(np.float32)

        label_index = LABEL_TO_INDEX[motion_name]
        generated = sample_motion(
            model,
            diffusion,
            label_index=label_index,
            n_samples=n_samples_per_label,
            device=device,
        ).astype(np.float32)

        fmd = frechet_distance(real, generated)
        mpjpe = mpjpe_nearest(real, generated)
        variance = motion_variance(generated)
        rows.append((motion_name, fmd, mpjpe, variance))

        print(
            f"{motion_name:>8} | FMD: {fmd:.6f} | MPJPE: {mpjpe:.6f} | Var: {variance:.6f}",
            flush=True,
        )

    write_results(rows)
    print(f'Results saved to {RESULTS_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate motion generation quality.')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--n-samples-per-label', type=int, default=128)
    args = parser.parse_args()

    evaluate(model_path=args.model_path, n_samples_per_label=args.n_samples_per_label)
