from model.unet import UNet
from utils.dataloader import DataLoader
from torch import optim
import torch.nn as nn
import torch


def train(model, device, data_path, epochs=40, batch_size=2, lr=0.00001):
    dataset = DataLoader(data_path)
    training_set = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for image, label in training_set:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = model(image)
            loss = criterion(pred, label)
            print("Train loss:", loss.item())

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'best_model.pth')

            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(1, 1)
    model.to(device=device)
    data_path = "data/train/"
    train(model, device, data_path=data_path)