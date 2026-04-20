import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from Src.Models.training_new import load_model, training_loop, save_model
from Src.Utils.data_utils import get_split_files
from Src.Utils.replicability import set_seed, make_worker_seed_fn, get_generator
from Src.Atlas.atlas_dataset_patch import AtlasDataset
from Src.Utils.image_utils import HistogramEqualizationd

import torch
from torch.utils.data import  DataLoader

from monai.transforms import (
    Compose, ScaleIntensityd, RandFlipd, RandRotate90d, RandZoomd,
    RandGaussianNoised, ToTensord,
    SpatialPadd, RandSpatialCropd
)



#---------------general configuration-------------------
MAX_FILES = 100 # pocet souboru pro testovani, pokud jich je vice, tak se pouzije jen prvnich MAX_FILES
START_FOLD = 4 # od ktereho foldu se zacne, 0 = vsechny foldy

config = {}
config["NAME"] = f"Atlas-heqv-multi-patch-{MAX_FILES}"
config["NUM_CLASSES"]=7 # 6 obratle + pozadí
config["BINARY"]=False # pokud je True, tak se upravi labely na 0,1 (0 = pozadí, 1 = objekt) v datasetu
config["MAX_EPOCHS"]=200 # pak asi 600
config["LR"] = 1e-2
config["TRAIN_BATCH_SIZE"] = 2
config["TEST_BATCH_SIZE"] =  2
config["TRAIN"] = True
config["LOAD_TRAIN_MODEL"] = None # f"../Models/{config['NAME']}-final"
config["SAVE_EPOCH"] = 10
config["MIN_VAL_DICE"] = 0.65
config["SEED"]= 42 # seed pro replicabilitu
lrconfig = {}
lrconfig["LRReduceOnPlato"] = False
# plati jen pro LRReduceonPlato
lrconfig["LR_PATIENCE"] = 5
lrconfig["LR_RATIO"] = 0.8
# plati jen pro CosineAnnealingWarmRestarts
lrconfig["T0"]=5
lrconfig["T_MULT"]=2
lrconfig["ETA_MIN"]=1e-8


def get_train_transformation():
    transforms = Compose([
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
    ])
    transforms.set_random_state(config["SEED"])
    return transforms


def get_test_transformation():
    return Compose([
        HistogramEqualizationd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(512, 512)),
        ToTensord(keys=["image", "label"]),
    ])

if __name__=="__main__":
   # set seed pro replicabilitu
   set_seed(config["SEED"]) # nastaveni seed pro replicabilitu
   worker_init_fn = make_worker_seed_fn(config["SEED"])
   seeded_generator = get_generator(config["SEED"])
   #==========================
   print("Atlas segmentation model")
   data_path = "data_Atlas-vertebra"
   folds_path = f"Atlas_vertebra_folds"
   images_path = os.path.join(data_path, "datasets-PNG")
   labels_path = os.path.join(data_path, "datasets-NPY")
   folds= get_split_files(images_path=images_path,
                          labels_path=labels_path,
                          folds_path=folds_path,
                          dataset_name="atlas_vertebra", max_files=MAX_FILES)
   for fold_id, fold in enumerate(folds[START_FOLD:], start=START_FOLD):
      print(f"Fold {fold_id}")
      print(f"Train images: {len(fold['train']['image_name'])}")
      print(f"Train labels: {len(fold['train']['label_name'])}")
      print(f"Val images: {len(fold['val']['image_name'])}")
      print(f"Val labels: {len(fold['val']['label_name'])}")
      
      train_transformation = get_train_transformation()
      val_transformation = get_test_transformation()

      train_dataset=AtlasDataset(images=fold['train']['image_path'],
                                masks=fold['train']['label_path'],
                                classes=config["NUM_CLASSES"],
                                image_transform=train_transformation,
                                binary=config["BINARY"], 
                                patches_per_image=20)
      train_loader = DataLoader(train_dataset,
                                batch_size=config["TRAIN_BATCH_SIZE"],
                                shuffle=True,
                                num_workers=0, # bylo 16, ale na 16 to padalo
                                pin_memory=torch.cuda.is_available(),
                                worker_init_fn=worker_init_fn,
                                generator=seeded_generator
                                )

      test_dataset=AtlasDataset(images=fold['val']['image_path'],
                               masks=fold['val']['label_path'],
                               classes=config["NUM_CLASSES"],
                               image_transform=val_transformation,
                                binary=config["BINARY"],)
      test_loader = DataLoader(test_dataset,
                                 batch_size=config["TEST_BATCH_SIZE"],
                                 shuffle=False,
                                 num_workers=0, # bylo 16, ale na 16 to padalo
                                 pin_memory=torch.cuda.is_available(),
                                 worker_init_fn=worker_init_fn,
                                 generator=seeded_generator
                               )
      
      x, y = next(iter(train_loader))
      print("image:", x.shape, x.min().item(), x.mean().item(), x.max().item())
      print("label:", y.shape, y.min().item(), y.max().item())
      print("label sum per-pixel (mean):", y.sum(dim=1).float().mean().item())
      print("label class occupancy:", y.sum(dim=(0,2,3)))

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      model = UNet(
          spatial_dims=2,  # Pro 2D segmentaci
          in_channels=3,  # Počet kanálů vstupního obrázku (RGB = 3)
          out_channels=config["NUM_CLASSES"],  # Počet tříd
          channels=(16, 32, 64, 128, 256),  # Počet kanálů v jednotlivých vrstvách
          strides=(2, 2, 2, 2),  # Stride pro downsampling
          num_res_units=2  # Počet reziduálních bloků
      ).to(device)
      loss_function = DiceCELoss(
            to_onehot_y=False, 
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0)
      optimizer = torch.optim.Adam(model.parameters(), config["LR"])
      if config["TRAIN"]:
          epoch = 0 # default initil epoch
          if config["LOAD_TRAIN_MODEL"] is not None:  # Pokud je zadáno jméno modelu, tak se načte
              # load stored model for next training
              model, optimizer, epoch = load_model(model, optimizer,
                                                   filepath=f"./Models/{config['LOAD_TRAIN_MODEL']}-fold-{fold_id}.pth",
                                                   device=device)
          # run training loop for fold
          training_loop(model=model, loss_function=loss_function,
                        optimizer=optimizer, train_loader=train_loader,
                        val_loader=test_loader,
                        config=config, lrconfig=lrconfig,
                        fold_id=fold_id, start_epoch=epoch,)

          save_model(model, optimizer, epoch=config["MAX_EPOCHS"],
                     filepath=f"./Models/{config['NAME']}-fold-{fold_id}-final.pth")



