import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class Sentinel2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        label = 1 if "glof" in img_path else 0
        if self.transform:
            image = self.transform(image)
        return image, label
