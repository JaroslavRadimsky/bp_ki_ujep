"""Training entry point for patch-based atlas vertebra segmentation.

This script assembles the full training pipeline:
1. resolve dataset and fold locations,
2. build MONAI transforms,
3. create training and validation datasets,
4. initialize the segmentation model and optimizer, and
5. run fold-based training with checkpointing.
"""

import os

import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    RandZoomd,
    ScaleIntensityd,
    SpatialPadd,
    ToTensord,
)
from torch.utils.data import DataLoader

from atlas_dataset_patch import AtlasDataset
from data_utils import get_split_files
from image_utils import HistogramEqualizationd
from replicability import get_generator, make_worker_seed_fn, set_seed
from training_new import load_model, save_model, training_loop


# --- Path resolution --------------------------------------------------------

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
DIGITECH_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "Digitech")


def resolve_existing_path(*parts):
    """Resolve a path first inside this repository and then inside Digitech.

    The function lets the script run from this thesis repository while still
    reusing data assets stored in the neighboring ``Digitech`` workspace.
    """
    local_path = os.path.join(PROJECT_ROOT, *parts)
    if os.path.exists(local_path):
        return local_path

    digitech_path = os.path.join(DIGITECH_ROOT, *parts)
    if os.path.exists(digitech_path):
        return digitech_path

    return local_path


# --- Experiment configuration ----------------------------------------------

MAX_FILES = 100
START_FOLD = 4

config = {}
config["NAME"] = f"Atlas-heqv-multi-patch-{MAX_FILES}"
config["NUM_CLASSES"] = 7          # Six vertebra classes plus background.
config["BINARY"] = False           # Clip labels into the configured class range.
config["MAX_EPOCHS"] = 200         # Main training budget for each fold.
config["LR"] = 1e-2                # Initial optimizer learning rate.
config["TRAIN_BATCH_SIZE"] = 2
config["TEST_BATCH_SIZE"] = 2
config["TRAIN"] = True
config["LOAD_TRAIN_MODEL"] = None  # Set to a checkpoint stem to resume training.
config["SAVE_EPOCH"] = 10          # Save periodic checkpoints every N epochs.
config["MIN_VAL_DICE"] = 0.65      # Minimum Dice required before saving "best".
config["SEED"] = 42

lrconfig = {}
lrconfig["LRReduceOnPlato"] = False
lrconfig["LR_PATIENCE"] = 5        # Used only by ReduceLROnPlateau.
lrconfig["LR_RATIO"] = 0.8         # Used only by ReduceLROnPlateau.
lrconfig["T0"] = 5                 # Used only by CosineAnnealingWarmRestarts.
lrconfig["T_MULT"] = 2             # Used only by CosineAnnealingWarmRestarts.
lrconfig["ETA_MIN"] = 1e-8         # Used only by CosineAnnealingWarmRestarts.


def get_train_transformation():
    """Create the stochastic transform pipeline used for training patches."""
    transforms = Compose(
        [
            HistogramEqualizationd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
            RandSpatialCropd(keys=["image", "label"], roi_size=(512, 512), random_size=False),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.2),
            ToTensord(keys=["image", "label"]),
        ]
    )
    transforms.set_random_state(config["SEED"])
    return transforms


def get_test_transformation():
    """Create the deterministic transform pipeline used for validation."""
    return Compose(
        [
            HistogramEqualizationd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
            ToTensord(keys=["image", "label"]),
        ]
    )


if __name__ == "__main__":
    # Seed every relevant source of randomness before creating datasets.
    set_seed(config["SEED"])
    worker_init_fn = make_worker_seed_fn(config["SEED"])
    seeded_generator = get_generator(config["SEED"])

    print("Atlas segmentation model")

    # Resolve dataset assets either locally or in the sibling Digitech workspace.
    data_path = resolve_existing_path("data_Atlas-vertebra")
    folds_path = resolve_existing_path("Atlas_vertebra_folds")
    images_path = os.path.join(data_path, "datasets-PNG")
    labels_path = os.path.join(data_path, "datasets-NPY")
    models_dir = os.path.join(PROJECT_ROOT, "Models")
    os.makedirs(models_dir, exist_ok=True)

    folds = get_split_files(
        images_path=images_path,
        labels_path=labels_path,
        folds_path=folds_path,
        dataset_name="atlas_vertebra",
        max_files=MAX_FILES,
    )

    for fold_id, fold in enumerate(folds[START_FOLD:], start=START_FOLD):
        print(f"Fold {fold_id}")
        print(f"Train images: {len(fold['train']['image_name'])}")
        print(f"Train labels: {len(fold['train']['label_name'])}")
        print(f"Val images: {len(fold['val']['image_name'])}")
        print(f"Val labels: {len(fold['val']['label_name'])}")

        train_transformation = get_train_transformation()
        val_transformation = get_test_transformation()

        # Training uses multiple random crops per source image for data diversity.
        train_dataset = AtlasDataset(
            images=fold["train"]["image_path"],
            masks=fold["train"]["label_path"],
            classes=config["NUM_CLASSES"],
            image_transform=train_transformation,
            binary=config["BINARY"],
            patches_per_image=20,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["TRAIN_BATCH_SIZE"],
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            generator=seeded_generator,
        )

        # Validation keeps one deterministic sample per image.
        test_dataset = AtlasDataset(
            images=fold["val"]["image_path"],
            masks=fold["val"]["label_path"],
            classes=config["NUM_CLASSES"],
            image_transform=val_transformation,
            binary=config["BINARY"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["TEST_BATCH_SIZE"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            generator=seeded_generator,
        )

        # Print a quick sanity check of the produced tensors before training.
        x, y = next(iter(train_loader))
        print("image:", x.shape, x.min().item(), x.mean().item(), x.max().item())
        print("label:", y.shape, y.min().item(), y.max().item())
        print("label sum per-pixel (mean):", y.sum(dim=1).float().mean().item())
        print("label class occupancy:", y.sum(dim=(0, 2, 3)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the 2D UNet used throughout the atlas segmentation experiments.
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=config["NUM_CLASSES"],
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

        # Combine Dice and cross-entropy to optimize overlap and class separation.
        loss_function = DiceCELoss(
            to_onehot_y=False,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), config["LR"])

        if config["TRAIN"]:
            epoch = 0

            # Resume from a stored checkpoint when requested by the config.
            if config["LOAD_TRAIN_MODEL"] is not None:
                model, optimizer, epoch = load_model(
                    model,
                    optimizer,
                    filepath=os.path.join(models_dir, f"{config['LOAD_TRAIN_MODEL']}-fold-{fold_id}.pth"),
                    device=device,
                )

            # Train and validate the current fold.
            training_loop(
                model=model,
                loss_function=loss_function,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=test_loader,
                config=config,
                lrconfig=lrconfig,
                fold_id=fold_id,
                start_epoch=epoch,
            )

            # Store the final checkpoint after the fold completes.
            save_model(
                model,
                optimizer,
                epoch=config["MAX_EPOCHS"],
                filepath=os.path.join(models_dir, f"{config['NAME']}-fold-{fold_id}-final.pth"),
            )
