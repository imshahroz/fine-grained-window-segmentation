import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SimpleSegNet
from dataset import WindowDataset


def train():
    image_dir = "data/images"
    mask_dir = "data/masks"

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError("Make sure data/images and data/masks exist.")

    dataset = WindowDataset(image_dir=image_dir, mask_dir=mask_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSegNet().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)
            loss = criterion(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/window_seg_model.pth")
    print("Model saved to outputs/window_seg_model.pth")


if __name__ == "__main__":
    train()
