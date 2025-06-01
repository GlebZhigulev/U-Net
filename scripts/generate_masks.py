import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm import tqdm
import sys

def create_mask(xml_file, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color=255, thickness=-1)
    except Exception as e:
        print(f"⚠️ Ошибка при обработке {xml_file}: {e}")

    return mask

def generate_masks(base_dir):
    images_dir = os.path.join(base_dir, 'images')
    annotations_dir = os.path.join(base_dir, 'annotations')
    masks_dir = os.path.join(base_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    for image_name in tqdm(image_files, desc=f'Генерация масок для {base_dir}'):
        base_name = image_name.replace('.jpg', '')
        image_path = os.path.join(images_dir, image_name)
        xml_path = os.path.join(annotations_dir, base_name + '.xml')
        mask_path = os.path.join(masks_dir, base_name + '.png')

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Пропущено: {image_path}")
            continue

        if not os.path.exists(xml_path):
            # Если XML нет, создаём полностью чёрную маску
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = create_mask(xml_path, image.shape)

        # Сохраняем даже пустые маски
        cv2.imwrite(mask_path, mask)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❌ Укажи путь к папке страны. Пример:")
        print("   py generate_masks.py data/China_Drone")
        sys.exit(1)

    generate_masks(sys.argv[1])
