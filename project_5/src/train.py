"""Train a DDPM to generate 48-frame stickman motion conditioned on a label."""
import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import MotionDataset
from diffusion import GaussianDiffusion
from model import MotionDenoiser


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'motion_diffusion.pth')


def evaluate_loss(model, diffusion, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for motion, labels in loader:
            motion = motion.to(device)
            labels = labels.to(device)
            timesteps = torch.randint(
                0, diffusion.n_steps, (motion.shape[0],), device=device, dtype=torch.long
            )
            noise = torch.randn_like(motion)
            noisy = diffusion.q_sample(motion, timesteps, noise=noise)
            predicted = model(noisy, timesteps, labels)
            total_loss += torch.nn.functional.mse_loss(predicted, noise).item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train(
    save_path=MODEL_PATH,
    n_epochs=200,
    batch_size=64,
    lr=2e-4,
    n_diffusion_steps=1000,
    log_interval=20,
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_dataset = MotionDataset(split='training')
    validation_dataset = MotionDataset(split='validation')

    weights = train_dataset.class_balanced_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    diffusion = GaussianDiffusion(n_steps=n_diffusion_steps, device=device)
    model = MotionDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}", flush=True)
    print(f"Training samples: {len(train_dataset)}, label counts: {train_dataset.label_counts()}", flush=True)
    print(f"Validation samples: {len(validation_dataset)}, label counts: {validation_dataset.label_counts()}", flush=True)
    print(f"Model parameters: {parameter_count}", flush=True)
    print(f"Batch size: {batch_size}, epochs: {n_epochs}, lr: {lr}, diffusion steps: {n_diffusion_steps}", flush=True)

    best_validation_loss = float('inf')
    model.train()

    for epoch in range(n_epochs):
        epoch_start = time.time()
        running_loss = 0.0

        for batch_index, (motion, labels) in enumerate(train_loader, start=1):
            motion = motion.to(device)
            labels = labels.to(device)

            timesteps = torch.randint(
                0, diffusion.n_steps, (motion.shape[0],), device=device, dtype=torch.long
            )
            noise = torch.randn_like(motion)
            noisy = diffusion.q_sample(motion, timesteps, noise=noise)
            predicted = model(noisy, timesteps, labels)
            loss = torch.nn.functional.mse_loss(predicted, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{n_epochs} | Batch {batch_index}/{len(train_loader)} "
                    f"| Avg Loss: {running_loss / batch_index:.6f}",
                    flush=True,
                )

        train_loss = running_loss / max(len(train_loader), 1)
        validation_loss = evaluate_loss(model, diffusion, validation_loader, device)
        message = (
            f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.6f}, "
            f"Val Loss: {validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'n_diffusion_steps': n_diffusion_steps,
                'sequence_length': 48,
                'num_joints': 15,
            }
            torch.save(checkpoint, save_path)
            message += ' [saved]'

        scheduler.step()
        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(f"{message}, Epoch Time: {elapsed:.1f}s, LR: {current_lr:.6f}", flush=True)

    print(f"Best model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train motion diffusion model.')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--diffusion-steps', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-path', type=str, default=MODEL_PATH)
    args = parser.parse_args()

    train(
        save_path=args.save_path,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_diffusion_steps=args.diffusion_steps,
        log_interval=args.log_interval,
        seed=args.seed,
    )
