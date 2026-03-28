import argparse
from datetime import datetime 
from pathlib import Path

import torch
from torch import nn
import numpy as np
from homework.datasets.road_dataset import load_data
from homework.models import load_model, save_model
from homework.metrics import AccuracyMetric
import torch.utils.tensorboard as tb    

def train_detection(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:      
        print("No GPU found, using CPU.")  
        device = torch.device("cpu")

    # set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)    

    # create experiment directory with timestamp to save logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # bake in kwargs for model loading
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # create dataloader for training set
    train_detection_dataset = load_data("drive_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)

    # create dataloader for validation set
    val_detection_dataset = load_data("drive_data/val", transform_pipeline="default", shuffle=False)

    # create loss function and optimizer
    seg_loss_func = torch.nn.CrossEntropyLoss()
    depth_loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create metric for evaluation
    global_step = 0
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()

    # training loop
    for epoch in range(num_epoch):

        # clear metrics at the start of each epoch
        train_metric.reset()
        
        model.train()   

        for batch in train_detection_dataset:
            img = batch["image"].to(device)
            seg_label = batch["track"].to(device)
            depth_label = batch["depth"].to(device)

            optimizer.zero_grad()
            logits, pred_depth = model(img)
            preds = logits.argmax(dim=1)
            train_metric.add(preds, seg_label)  
            seg_loss = seg_loss_func(logits, seg_label)
            depth_loss = depth_loss_func(pred_depth, depth_label)
            loss = seg_loss + 0.1 * depth_loss
            loss.backward()
            optimizer.step()

            # log to tensorboard
            logger.add_scalar("train/loss", loss.item(), global_step)
            logger.add_scalar("train/seg_loss", seg_loss.item(), global_step)
            logger.add_scalar("train/depth_loss", depth_loss.item(), global_step)

            global_step += 1

        # evaluate on validation set at the end of each epoch
        val_metric.reset()
        with torch.inference_mode():
            model.eval()

            for batch in train_detection_dataset:
                img = batch["image"].to(device)
                seg_label = batch["track"].to(device)
                depth_label = batch["depth"].to(device)

                logits = model(img)
                preds = logits.argmax(dim=1)
                val_metric.add(preds, seg_label)

            train_acc = train_metric.compute()["accuracy"]
            val_acc = val_metric.compute()["accuracy"]

            logger.add_scalar("train/accuracy", train_acc, epoch)
            logger.add_scalar("val/accuracy", val_acc, epoch)

            # print progress
            if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                    f"train_acc={train_acc:.4f} "
                    f"val_acc={val_acc:.4f}"
                )

    # save the final model checkpoint
    save_model(model)

    # save a copy of the model checkpoint in the experiment directory for this run
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Saved model checkpoint to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for road detection")
    parser.add_argument("--exp_dir", type=str, default="logs", help="Directory to save logs and model checkpoints")
    parser.add_argument("--model_name", type=str, default="detector", help="Model name for saving and loading")
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")

    args = parser.parse_args()
    train_detection(**vars(args))

