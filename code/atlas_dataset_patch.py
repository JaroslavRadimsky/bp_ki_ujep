import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class AtlasDataset(Dataset):
    def __init__(self, images, masks, image_transform=None, classes=2, binary=True, patches_per_image=1):
        self.image_transform = image_transform
        self.image_filenames = images
        self.label_filenames = masks
        self.classes = classes
        self.binary = binary
        self.patches_per_image = patches_per_image

    def __len__(self):
        return len(self.image_filenames) * self.patches_per_image

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_filenames)

        img_path = self.image_filenames[real_idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))

        label_path = self.label_filenames[real_idx]
        label = np.load(label_path)

        if self.binary:
            label = np.clip(label, 0, self.classes - 1)

        if len(label.shape) == 2:
            label = np.eye(self.classes)[label].transpose(2, 0, 1)

        data = {"image": image, "label": label}
        if self.image_transform is not None:
            data = self.image_transform(data)

        return data["image"], data["label"]
