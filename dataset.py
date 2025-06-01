import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadDefectDataset(Dataset):
    def __init__(self, images_dir, masks_dir, split_file, transform=None):
        with open(split_file, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        image_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.masks_dir, name + ".png")

        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"❌ Не удалось загрузить изображение: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Загрузка маски
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"❌ Не удалось загрузить маску: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        # Аугментации
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
