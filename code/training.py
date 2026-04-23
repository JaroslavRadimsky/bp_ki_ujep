"""Training utilities for vertebra segmentation models.

The functions in this module implement checkpoint persistence and the training
loop shared by the atlas segmentation experiments.
"""

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


def save_model(model, optimizer, epoch, filepath="../Models/model.pth"):
    """Save model and optimizer state into a checkpoint file."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Saved model to: {filepath}")


def load_model(model, optimizer, filepath="model.pth", device="cuda"):
    """Load model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch


def training_loop(model, loss_function, optimizer, train_loader, val_loader, config, lrconfig, fold_id, start_epoch=0):
    """Run the full training and validation loop for one fold.

    The loop tracks loss, Dice score, learning rate, and periodically stores
    both regular and best-performing checkpoints.
    """
    min_val_dice = config["MIN_VAL_DICE"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter(f'./Runs/{config["NAME"]}-fold-{fold_id}')

    # Excluding the background class gives a more informative foreground Dice.
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    use_plateau = lrconfig["LRReduceOnPlato"]
    if use_plateau:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lrconfig["LR_RATIO"],
            patience=lrconfig["LR_PATIENCE"],
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=lrconfig["T0"],
            T_mult=lrconfig["T_MULT"],
            eta_min=lrconfig["ETA_MIN"],
        )

    best_val_dice = 0.0

    for epoch in range(start_epoch, config["MAX_EPOCHS"]):
        # --- Training phase -------------------------------------------------
        model.train()
        epoch_loss = 0.0
        total_batches = 0
        dice_metric.reset()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

            # Convert logits to one-hot predictions before Dice evaluation.
            pred_class = torch.argmax(logits, dim=1)
            pred_onehot = F.one_hot(pred_class, num_classes=7).permute(0, 3, 1, 2).float()
            dice_metric(pred_onehot, labels)

        train_epoch_dice = float(dice_metric.aggregate().item())
        epoch_loss = epoch_loss / max(1, total_batches)

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Dice/train", train_epoch_dice, epoch)

        # --- Validation phase -----------------------------------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0
        dice_metric.reset()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Validation uses sliding-window inference to mirror test-time behavior.
                logits = sliding_window_inference(
                    inputs=inputs,
                    roi_size=(512, 512),
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.25,
                    mode="gaussian",
                )

                loss = loss_function(logits, labels)
                val_loss += loss.item()
                val_batches += 1

                pred_class = torch.argmax(logits, dim=1)
                pred_onehot = F.one_hot(pred_class, num_classes=7).permute(0, 3, 1, 2).float()
                dice_metric(pred_onehot, labels)

        epoch_val_dice = float(dice_metric.aggregate().item())
        val_loss = val_loss / max(1, val_batches)

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val", epoch_val_dice, epoch)

        # Update the chosen scheduler and log the effective learning rate.
        if use_plateau:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            scheduler.step(epoch)
            current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("LR", current_lr, epoch)

        # Save periodic checkpoints so interrupted runs can be resumed.
        if epoch % config["SAVE_EPOCH"] == 0:
            save_model(model, optimizer, epoch, filepath=f"./Models/{config['NAME']}-epoch-{epoch}-fold-{fold_id}.pth")

        # Save a best checkpoint only after the configured validation threshold is met.
        if epoch_val_dice > best_val_dice and epoch_val_dice > min_val_dice:
            best_val_dice = epoch_val_dice
            save_model(model, optimizer, epoch, filepath=f"./Models/{config['NAME']}-fold-{fold_id}-best.pth")

        print(
            f"Epoch {epoch}: "
            f"Loss {epoch_loss:.4f} | Train Dice {train_epoch_dice:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Dice {epoch_val_dice:.4f} | LR {current_lr:.6f}"
        )

    writer.close()
