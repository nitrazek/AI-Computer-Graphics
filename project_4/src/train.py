import os
import time
import torch
from torch.utils.data import DataLoader

from dataset import ShapeFlowDataset
from model import VectorFieldNet
from helpers import chamfer_distance_torch


def evaluate_loss(model, dataloader, device, n_steps):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for source_points, target_points in dataloader:
            source_points = source_points.to(device)
            target_points = target_points.to(device)
            predicted = model(source_points, n_steps=n_steps)
            running_loss += chamfer_distance_torch(predicted, target_points).item()

    model.train()
    return running_loss / max(len(dataloader), 1)


def train_model(
    source_obj_path,
    target_obj_path,
    save_path,
    n_source_points=1024,
    n_target_points=1024,
    samples_per_epoch=512,
    validation_samples=64,
    n_epoch=40,
    batch_size=8,
    lr=1e-3,
    n_integration_steps=8,
    log_interval=10,
    seed=0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_dataset = ShapeFlowDataset(
        source_obj_path=source_obj_path,
        target_obj_path=target_obj_path,
        n_source_points=n_source_points,
        n_target_points=n_target_points,
        samples_per_epoch=samples_per_epoch,
        augment=True,
    )
    validation_dataset = ShapeFlowDataset(
        source_obj_path=source_obj_path,
        target_obj_path=target_obj_path,
        n_source_points=n_source_points,
        n_target_points=n_target_points,
        samples_per_epoch=validation_samples,
        augment=True,
        seed=seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = VectorFieldNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    best_validation_loss = float('inf')
    model_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"Starting training for: {save_path}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Source: {source_obj_path}, Target: {target_obj_path}", flush=True)
    print(f"Training samples per epoch: {len(train_dataset)}, batches: {len(train_loader)}", flush=True)
    print(f"Validation samples: {len(validation_dataset)}, batches: {len(validation_loader)}", flush=True)
    print(f"Model parameters: {model_parameters}", flush=True)
    print(f"Batch size: {batch_size}, epochs: {n_epoch}, learning rate: {lr}", flush=True)
    print(f"Source points: {n_source_points}, target points: {n_target_points}, integration steps: {n_integration_steps}", flush=True)

    model.train()

    for epoch in range(n_epoch):
        running_loss = 0.0
        epoch_start_time = time.time()
        print(f"Epoch {epoch + 1}/{n_epoch} started", flush=True)

        for batch_index, (source_points, target_points) in enumerate(train_loader, start=1):
            batch_start_time = time.time()
            source_points = source_points.to(device)
            target_points = target_points.to(device)

            predicted = model(source_points, n_steps=n_integration_steps)
            loss = chamfer_distance_torch(predicted, target_points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(train_loader):
                average_batch_loss = running_loss / batch_index
                batch_elapsed = time.time() - batch_start_time
                print(
                    f"Epoch {epoch + 1}/{n_epoch} | Batch {batch_index}/{len(train_loader)} | "
                    f"Avg Loss: {average_batch_loss:.6f} | Step Time: {batch_elapsed:.2f}s",
                    flush=True,
                )

        train_loss = running_loss / max(len(train_loader), 1)
        validation_loss = evaluate_loss(model, validation_loader, device, n_integration_steps)
        message = f"Epoch {epoch + 1}/{n_epoch}, Train Loss: {train_loss:.6f}, Val Loss: {validation_loss:.6f}"

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), save_path)
            message += ' [saved]'

        scheduler.step()
        elapsed = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f"{message}, Epoch Time: {elapsed:.1f}s, LR: {current_lr:.6f}", flush=True)

    print(f"Model saved to {save_path}.")


if __name__ == '__main__':
    source_objects = [
        ('bunny', '../models/bunny.obj'),
        ('dragon', '../models/dragon_small.obj'),
        ('armadillo', '../models/armadillo_small.obj'),
    ]
    target_obj = '../models/teapot.obj'

    for name, source_path in source_objects:
        train_model(
            source_obj_path=source_path,
            target_obj_path=target_obj,
            save_path=f'../models/{name}_flow.pth',
        )
