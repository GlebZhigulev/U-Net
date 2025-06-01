from dataset import RoadDefectDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = RoadDefectDataset(
    images_dir='data/Czech/images',
    masks_dir='data/Czech/masks',
    split_file='data/splits/test.txt'
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, masks in loader:
    for i in range(4):
        img = images[i].permute(1, 2, 0).numpy()
        msk = masks[i][0].numpy()

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Image')

        plt.subplot(1, 2, 2)
        plt.imshow(msk, cmap='gray')
        plt.title('Mask')

        plt.show()
    break
