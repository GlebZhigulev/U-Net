import os
import random

IMAGES_DIR = 'data/images'
SPLITS_DIR = 'data/splits'

os.makedirs(SPLITS_DIR, exist_ok=True)

# Получаем список всех изображений без расширения
all_images = [f.replace('.jpg', '') for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]

# Перемешиваем с фиксированным seed для воспроизводимости
random.seed(42)
random.shuffle(all_images)

# Разделим: 70% — train, 15% — val, 15% — test
total = len(all_images)
train_split = int(0.7 * total)
val_split = int(0.85 * total)

train_files = all_images[:train_split]
val_files = all_images[train_split:val_split]
test_files = all_images[val_split:]

# Функция сохранения списка файлов
def save_split(filename, split):
    with open(os.path.join(SPLITS_DIR, filename), 'w') as f:
        for name in split:
            f.write(name + '\n')

save_split('train.txt', train_files)
save_split('val.txt', val_files)
save_split('test.txt', test_files)

print(f'Всего изображений: {total}')
print(f'Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}')
