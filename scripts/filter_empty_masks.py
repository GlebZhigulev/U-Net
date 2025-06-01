import os

splits = ['train', 'val', 'test']
masks_dir = 'data/masks'

for split in splits:
    split_path = f'data/splits/{split}.txt'
    with open(split_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    filtered = [name for name in lines if os.path.exists(os.path.join(masks_dir, name + '.png'))]

    with open(split_path, 'w') as f:
        for name in filtered:
            f.write(name + '\n')

    print(f"{split}: удалено {len(lines) - len(filtered)} пустых записей, осталось {len(filtered)}")
