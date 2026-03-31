"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

# citation: used copilot and chatgpt for guidance

import argparse
from datetime import datetime
from pathlib import Path
from random import seed

import torch
import numpy as np
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from torch import nn, optim
from homework.metrics import PlannerMetric
from homework.models import MLPPlanner, TransformerPlanner, load_model, save_model
from homework.datasets.road_dataset import load_data 
import torch.utils.tensorboard as tb

def train(
    exp_dir: str = "logs",
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=4,
    lr=1e-3,
    batch_size=128,
    num_epoch=40,
    **kwargs,
): 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("No GPU found, using CPU")
        device = torch.device("cpu")
    
    # set random seed for reproducibility
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    # create experiment directory with timestamp to save logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # bake in kwargs for model loading
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # create dataloader for training
    train_data = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        return_dataloader=True,
    )

    # create dataloader for validation
    val_data = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        shuffle=False,
    )

    # creare loss function and metric
    loss = nn.L1Loss(reduction="none")
    metric = PlannerMetric()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    for epoch in range(num_epoch):
        for batch in train_data:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(
                track_left=batch["track_left"],
                track_right=batch["track_right"],
            )
            loss_masked = loss(preds, batch["labels"]) * batch["labels_mask"][..., None]
            loss_mean = loss_masked.sum() / batch["labels_mask"].sum()

            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            logger.add_scalar("train/loss", loss_mean.item(), global_step)
            global_step += 1

        # compute validation metric at the end of each epoch
        metric.reset()
        with torch.inference_mode():
            for batch in val_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(batch)
                metric.add(preds.cpu(), batch["labels"].cpu(), batch["labels_mask"].cpu())

        val_metrics = metric.compute()
        logger.add_scalar("val/l1_error", val_metrics["l1_error"], epoch)
        logger.add_scalar("val/longitudinal_error", val_metrics["longitudinal_error"], epoch)
        logger.add_scalar("val/lateral_error", val_metrics["lateral_error"], epoch)

        # print progress
        print(
            f"Epoch {epoch+1}/{num_epoch} - "
            f"Train Loss: {loss_mean.item():.4f} - "
            f"Val L1 Error: {val_metrics['l1_error']:.4f} - "
            f"Val Longitudinal Error: {val_metrics['longitudinal_error']:.4f} - "
            f"Val Lateral Error: {val_metrics['lateral_error']:.4f}"
        )
    
    # save the final model checkpoint
    save_model(model)
    
    # save a copy of the mode; checkpoint in the experiment directory for easy reference
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model checkpoint saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="linear_planner")
    parser.add_argument("--transform_pipeline", type=str, default="state_only")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=40)

    args = parser.parse_args()
    train(**vars(args))

