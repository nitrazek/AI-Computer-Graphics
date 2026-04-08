import os
import time
import torch
from torch.utils.data import DataLoader

from dataset import ExposureDataset
from model import ExposureSynthesisCNN


def compute_loss(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, under_targets, over_targets, _ in dataloader:
            inputs = inputs.to(device)
            under_targets = under_targets.to(device)
            over_targets = over_targets.to(device)
            under_outputs, over_outputs = model(inputs)
            loss = loss_fn(under_outputs, under_targets) + loss_fn(over_outputs, over_targets)
            running_loss += loss.item()

    model.train()
    return running_loss / max(len(dataloader), 1)


def train_model(
    train_metadata_path,
    validation_metadata_path=None,
    data_root='../data',
    data_size=None,
    validation_size=None,
    crop_size=256,
    patches_per_image=8,
    validation_patches_per_image=4,
    n_epoch=50,
    batch_size=4,
    lr=0.001,
    save_path='../models/exposure_synthesis_model.pth',
    log_interval=5,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_dataset = ExposureDataset(
        train_metadata_path,
        data_root=data_root,
        data_size=data_size,
        crop_size=crop_size,
        patches_per_image=patches_per_image,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_loader = None
    if validation_metadata_path is not None:
        validation_dataset = ExposureDataset(
            validation_metadata_path,
            data_root=data_root,
            data_size=validation_size,
            crop_size=crop_size,
            patches_per_image=validation_patches_per_image,
        )
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = ExposureSynthesisCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.01)
    best_validation_loss = float('inf')
    model_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f'Starting training for: {save_path}', flush=True)
    print(f'Device: {device}', flush=True)
    print(f'Training samples: {len(train_dataset)}, batches per epoch: {len(train_loader)}', flush=True)
    if validation_loader is not None:
        print(f'Validation samples: {len(validation_dataset)}, validation batches: {len(validation_loader)}', flush=True)
    print(f'Model parameters: {model_parameters}', flush=True)
    print(f'Batch size: {batch_size}, epochs: {n_epoch}, learning rate: {lr}', flush=True)
    print(f'Crop size: {crop_size}, training patches per image: {patches_per_image}', flush=True)

    model.train()

    for epoch in range(n_epoch):
        running_loss = 0.0
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{n_epoch} started', flush=True)

        for batch_index, (inputs, under_targets, over_targets, _) in enumerate(train_loader, start=1):
            batch_start_time = time.time()
            inputs = inputs.to(device)
            under_targets = under_targets.to(device)
            over_targets = over_targets.to(device)

            under_outputs, over_outputs = model(inputs)
            loss = loss_fn(under_outputs, under_targets) + loss_fn(over_outputs, over_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(train_loader):
                average_batch_loss = running_loss / batch_index
                batch_elapsed = time.time() - batch_start_time
                print(
                    f'Epoch {epoch + 1}/{n_epoch} | Batch {batch_index}/{len(train_loader)} | '
                    f'Avg Loss: {average_batch_loss:.6f} | Step Time: {batch_elapsed:.2f}s',
                    flush=True,
                )

        train_loss = running_loss / max(len(train_loader), 1)
        message = f'Epoch {epoch + 1}/{n_epoch}, Train Loss: {train_loss:.6f}'

        if validation_loader is not None:
            validation_loss = compute_loss(model, validation_loader, loss_fn, device)
            message += f', Val Loss: {validation_loss:.6f}'

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), save_path)
                message += ' [saved]'
        else:
            torch.save(model.state_dict(), save_path)

        scheduler.step()
        elapsed = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f'{message}, Epoch Time: {elapsed:.1f}s, LR: {current_lr:.6f}', flush=True)

    print(f'Model saved to {save_path}.')


if __name__ == '__main__':
    train_model(
        '../data/training/metadata.csv',
        validation_metadata_path='../data/validation/metadata.csv',
        save_path='../models/exposure_synthesis_model.pth',
    )