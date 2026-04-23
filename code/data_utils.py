"""Utilities for loading predefined train/validation dataset splits.

The fold CSV files describe which image and label belong to each split. This
module turns those CSV rows into a structure that is easy to consume from the
training scripts.
"""

import os

import pandas as pd


def create_fold_info():
    """Create the canonical nested dictionary used for one fold."""
    return {
        "train": {
            "image_name": [],
            "image_path": [],
            "label_name": [],
            "label_path": [],
        },
        "val": {
            "image_name": [],
            "image_path": [],
            "label_name": [],
            "label_path": [],
        },
    }


def fill_fold_info(fold_info, fold_type, image_name, label_name, images_path, labels_path):
    """Append one image-label pair to the requested fold split.

    The function stores both file names and fully resolved paths so downstream
    code can report readable names while still having direct access to disk
    locations.
    """
    fold_info[fold_type]["image_name"].append(image_name)
    fold_info[fold_type]["label_name"].append(label_name)
    fold_info[fold_type]["image_path"].append(os.path.join(images_path, image_name))
    fold_info[fold_type]["label_path"].append(os.path.join(labels_path, label_name))

    # Print missing-file warnings early so training failures are easier to trace.
    if not os.path.exists(fold_info[fold_type]["image_path"][-1]):
        print(f"Image {fold_info[fold_type]['image_path'][-1]} does not exist.")

    if not os.path.exists(fold_info[fold_type]["label_path"][-1]):
        print(f"Label {fold_info[fold_type]['label_path'][-1]} does not exist.")


def get_split_files(dataset_name=None, images_path=None, labels_path=None, folds_path=None, k=5, max_files=1_000_000):
    """Load train/validation folds from CSV files.

    Args:
        dataset_name: Dataset identifier embedded in the fold CSV filenames.
        images_path: Root directory containing input images.
        labels_path: Root directory containing mask arrays.
        folds_path: Directory with fold CSV files.
        k: Number of folds to scan.
        max_files: Optional upper bound per split for quicker experiments.

    Returns:
        A list of fold dictionaries created by :func:`create_fold_info`.
    """
    assert dataset_name is not None, "dataset_name must be provided"
    assert images_path is not None, "images_path must be provided"
    assert labels_path is not None, "labels_path must be provided"
    assert folds_path is not None, "folds_path must be provided"

    splits_files = []
    if max_files is None:
        max_files = 1_000_000

    for i in range(k):
        train_fold_name = f"train_{dataset_name}_fold_{i}.csv"
        val_fold_name = f"val_{dataset_name}_fold_{i}.csv"
        train_fold_path = os.path.join(folds_path, train_fold_name)
        val_fold_path = os.path.join(folds_path, val_fold_name)

        # Skip missing folds instead of failing immediately so partial setups can
        # still be inspected.
        if not os.path.exists(train_fold_path):
            print(os.getcwd())
            print(f"Train fold {train_fold_name} does not exist")
            continue

        df_train_fold = pd.read_csv(train_fold_path)
        df_val_fold = pd.read_csv(val_fold_path)
        fold_info = create_fold_info()

        # Populate the training split with fully resolved image/label paths.
        for index, row in df_train_fold.iterrows():
            image_name = row["image"]
            label_name = row["label"]
            fill_fold_info(fold_info, "train", image_name, label_name, images_path, labels_path)
            if index >= max_files - 1:
                break

        # Populate the validation split with fully resolved image/label paths.
        for index, row in df_val_fold.iterrows():
            image_name = row["image"]
            label_name = row["label"]
            fill_fold_info(fold_info, "val", image_name, label_name, images_path, labels_path)
            if index >= max_files - 1:
                break

        splits_files.append(fold_info)

    return splits_files
