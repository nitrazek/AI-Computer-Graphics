import os
import time
import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model import ImageRestorationCNN

def evaluate_loss(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            running_loss += loss_fn(outputs, targets).item()

    model.train()
    return running_loss / max(len(dataloader), 1)


def train_model(
    input_paths,
    target_paths,
    data_size=None,
    n_epoch=25,
    batch_size=16,
    lr=0.001,
    save_path='../models/model.pth',
    validation_input_paths=None,
    validation_target_paths=None,
    validation_size=None,
    log_interval=10,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = ImageDataset(input_paths=input_paths, target_paths=target_paths, data_size=data_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    validation_loader = None
    if validation_input_paths and validation_target_paths:
        validation_dataset = ImageDataset(
            input_paths=validation_input_paths,
            target_paths=validation_target_paths,
            data_size=validation_size,
        )
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = ImageRestorationCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = torch.nn.L1Loss()
    best_validation_loss = float('inf')
    model_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"Starting training for: {save_path}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Training samples: {len(dataset)}, batches per epoch: {len(dataloader)}", flush=True)
    if validation_loader is not None:
        print(f"Validation samples: {len(validation_dataset)}, validation batches: {len(validation_loader)}", flush=True)
    print(f"Model parameters: {model_parameters}", flush=True)
    print(f"Batch size: {batch_size}, epochs: {n_epoch}, learning rate: {lr}", flush=True)

    model.train()

    for epoch in range(n_epoch):
        running_loss = 0.0
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{n_epoch} started", flush=True)

        for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
            batch_start_time = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(dataloader):
                average_batch_loss = running_loss / batch_index
                batch_elapsed = time.time() - batch_start_time
                print(
                    f"Epoch {epoch+1}/{n_epoch} | Batch {batch_index}/{len(dataloader)} | "
                    f"Avg Loss: {average_batch_loss:.6f} | Step Time: {batch_elapsed:.2f}s",
                    flush=True,
                )

        train_loss = running_loss / len(dataloader)
        message = f"Epoch {epoch+1}/{n_epoch}, Train Loss: {train_loss:.6f}"

        if validation_loader is not None:
            validation_loss = evaluate_loss(model, validation_loader, loss_fn, device)
            message += f", Val Loss: {validation_loss:.6f}"

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), save_path)
                message += ' [saved]'
        else:
            torch.save(model.state_dict(), save_path)

        scheduler.step()
        elapsed = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f"{message}, Epoch Time: {elapsed:.1f}s, LR: {current_lr:.6f}", flush=True)

    print(f"Model saved to {save_path}.")

if __name__ == "__main__":
    train_model(
        '../data/training/noisy_001',
        '../data/training/clean',
        data_size=1000,
        save_path='../models/denoising_001_model.pth',
        validation_input_paths='../data/validation/noisy_001',
        validation_target_paths='../data/validation/clean',
        validation_size=50,
    )
    train_model(
        '../data/training/noisy_003',
        '../data/training/clean',
        data_size=1000,
        save_path='../models/denoising_003_model.pth',
        validation_input_paths='../data/validation/noisy_003',
        validation_target_paths='../data/validation/clean',
        validation_size=50,
    )
    train_model(
        '../data/training/blurred_3',
        '../data/training/clean',
        data_size=1000,
        save_path='../models/deblurring_3_model.pth',
        validation_input_paths='../data/validation/blurred_3',
        validation_target_paths='../data/validation/clean',
        validation_size=50,
    )
    train_model(
        '../data/training/blurred_5',
        '../data/training/clean',
        data_size=1000,
        save_path='../models/deblurring_5_model.pth',
        validation_input_paths='../data/validation/blurred_5',
        validation_target_paths='../data/validation/clean',
        validation_size=50,
    )
