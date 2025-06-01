import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib as plt

from dataset import RoadDefectDataset
from models.unet import UNet

import os

# Параметры
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Датасеты
train_dataset = RoadDefectDataset(
    images_dir='data/images',
    masks_dir='data/masks',
    split_file='data/splits/train.txt'
)
val_dataset = RoadDefectDataset(
    images_dir='data/images',
    masks_dir='data/masks',
    split_file='data/splits/val.txt'
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# Модель
model = UNet(in_channels=3, out_channels=1).to(DEVICE)


criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses = []
val_losses = []

# Обучение
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)    
    print(f"[{epoch+1}/{EPOCHS}] Train Loss: {train_loss / len(train_loader):.4f}")

    # Валидация
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"           Val Loss: {val_loss / len(val_loader):.4f}")

# Сохранение
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/unet.pth')
print("✅ Модель сохранена в checkpoints/unet.pth")

 # График
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("UNet: динамика обучения")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/loss_curve.png")
plt.show()
