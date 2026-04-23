"""Dataset utilities for atlas vertebra segmentation experiments.

This module provides a lightweight PyTorch dataset that:
1. loads RGB atlas images from disk,
2. loads segmentation labels stored as NumPy arrays,
3. optionally converts labels to a clipped binary-style range, and
4. returns transformed image/label tensors ready for training.
"""

from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class AtlasDataset(Dataset):
    """PyTorch dataset for atlas vertebra images and segmentation masks.

    The dataset supports repeated sampling from the same source image through
    ``patches_per_image``. This is useful when random spatial cropping is
    performed inside the transform pipeline and multiple distinct crops are
    expected from a single original image.
    """

    def __init__(self, images, masks, image_transform=None, classes=2, binary=True, patches_per_image=1):
        """Store dataset metadata and preprocessing options.

        Args:
            images: Sequence of image file paths.
            masks: Sequence of mask file paths aligned with ``images``.
            image_transform: Optional MONAI-style transform applied to both items.
            classes: Number of output classes expected in the one-hot mask.
            binary: Whether to clip labels into the valid class range.
            patches_per_image: Virtual repetition factor for patch-based training.
        """
        self.image_transform = image_transform
        self.image_filenames = images
        self.label_filenames = masks
        self.classes = classes
        self.binary = binary
        self.patches_per_image = patches_per_image

    def __len__(self):
        """Return the virtual dataset length used by the DataLoader."""
        return len(self.image_filenames) * self.patches_per_image

    def __getitem__(self, idx):
        """Load one image-mask pair and apply the shared transform pipeline.

        The virtual index is mapped back to a real file index so that one image
        can be sampled multiple times with different random augmentations.
        """
        # Map the virtual patch index back to the original source image index.
        real_idx = idx % len(self.image_filenames)

        # Load the RGB image and convert it to channel-first float32 format.
        img_path = self.image_filenames[real_idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))

        # Load the segmentation label stored as a NumPy array.
        label_path = self.label_filenames[real_idx]
        label = np.load(label_path)

        # Optionally clamp values to the supported class range.
        if self.binary:
            label = np.clip(label, 0, self.classes - 1)

        # Convert 2D class indices to a channel-first one-hot representation.
        if len(label.shape) == 2:
            label = np.eye(self.classes)[label].transpose(2, 0, 1)

        # Apply the shared transform pipeline to both image and label together.
        data = {"image": image, "label": label}
        if self.image_transform is not None:
            data = self.image_transform(data)

        return data["image"], data["label"]
