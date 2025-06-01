import torch
import numpy as np
from torch.utils.data import DataLoader
from models.unet import UNet
from dataset import RoadDefectDataset
import matplotlib.pyplot as plt

# Метрики
def compute_iou(pred, target):
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()
    return intersection / union if union != 0 else 1.0

def compute_dice(pred, target):
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).sum().item()
    return (2 * intersection) / (pred.sum().item() + target.sum().item()) if (pred.sum() + target.sum()) != 0 else 1.0

# Устройство и модель
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/unet.pth", map_location=DEVICE))
model.eval()

# Данные
test_dataset = RoadDefectDataset(
    images_dir='data/images',
    masks_dir='data/masks',
    split_file='data/splits/test.txt'
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Метрики
ious = []
dices = []

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        preds = (preds > 0.5).float()

        iou = compute_iou(preds[0][0], masks[0][0])
        dice = compute_dice(preds[0][0], masks[0][0])

        ious.append(iou)
        dices.append(dice)

# Вывод
print(f"📊 Средний IoU: {np.mean(ious):.4f}")
print(f"📊 Средний Dice: {np.mean(dices):.4f}")

# После цикла с метриками (или вместо него — для отладки)
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        preds = (preds > 0.5).float()

        # Визуализация одного примера
        image_np = images[0].cpu().permute(1, 2, 0).numpy()
        mask_gt = masks[0][0].cpu().numpy()
        mask_pred = preds[0][0].cpu().numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title("🔹 Исходное изображение")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_gt, cmap='gray')
        plt.title("✅ Истинная маска")

        plt.subplot(1, 3, 3)
        plt.imshow(mask_pred, cmap='gray')
        plt.title("🧠 Предсказанная маска")

        plt.tight_layout()
        plt.show()
          # только один пример