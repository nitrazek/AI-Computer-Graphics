"""Inference helpers for motion diffusion model."""
import os
import torch

from diffusion import GaussianDiffusion
from model import MotionDenoiser
from skeleton import LABEL_TO_INDEX, SEQUENCE_LENGTH, NUM_JOINTS


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'motion_diffusion.pth'
)


def _normalize_label(prompt_text):
    text = prompt_text.lower().strip()
    if 'walk' in text:
        return 'walk'
    if 'jump' in text:
        return 'jump'
    raise ValueError("Prompt must contain either 'walk' or 'jump'.")


def label_from_prompt(prompt_text):
    label_name = _normalize_label(prompt_text)
    return label_name, LABEL_TO_INDEX[label_name]


def load_model_and_diffusion(model_path=DEFAULT_MODEL_PATH, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)
    n_diffusion_steps = 1000
    model_state = checkpoint

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        n_diffusion_steps = int(checkpoint.get('n_diffusion_steps', 1000))

    model = MotionDenoiser().to(device)
    model.load_state_dict(model_state)
    model.eval()

    diffusion = GaussianDiffusion(n_steps=n_diffusion_steps, device=device)
    return model, diffusion, device


@torch.no_grad()
def sample_motion(model, diffusion, label_index, n_samples=1, device=None):
    if device is None:
        device = next(model.parameters()).device
    labels = torch.full((n_samples,), int(label_index), device=device, dtype=torch.long)
    shape = (n_samples, SEQUENCE_LENGTH, NUM_JOINTS, 3)
    generated = diffusion.p_sample_loop(model, shape=shape, labels=labels, device=device)
    return generated.cpu().numpy()
