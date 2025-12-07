import os
from PIL import Image
from torch.utils.data import Dataset

class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # cat 文件夹
        cat_dir = os.path.join(root_dir, "cat")
        if os.path.isdir(cat_dir):
            for img in os.listdir(cat_dir):
                self.samples.append((os.path.join(cat_dir, img), 0))  # 猫=0

        # dog 文件夹
        dog_dir = os.path.join(root_dir, "dog")
        if os.path.isdir(dog_dir):
            for img in os.listdir(dog_dir):
                self.samples.append((os.path.join(dog_dir, img), 1))  # 狗=1

        print(f"加载数据：共 {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
