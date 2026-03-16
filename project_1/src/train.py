import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model import ImageRestorationCNN

def train_model(input_paths, target_paths, data_size, n_epoch=15, batch_size=10, lr=0.001, save_path='../model/model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = ImageDataset(input_paths=input_paths, target_paths=target_paths, data_size=data_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ImageRestorationCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()

    for epoch in range(n_epoch):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {running_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.")

if __name__ == "__main__":
    train_model('../data/training/noisy_001', '../data/training/clean', 200, save_path='../models/denoising_model.pth')
    train_model('../data/training/blurred_3', '../data/training/clean', 200, save_path='../models/deblurring_model.pth')