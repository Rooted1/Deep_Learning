import argparse
from datetime import datetime 
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from homework import metrics
from homework.datasets.classification_dataset import load_data
from homework.models import Classifier, load_model, save_model
from homework.metrics import AccuracyMetric
import torch.utils.tensorboard as tb

def train_classification(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
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
    train_classification_dataset = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)

    # create dataloader for validation set
    val_classification_dataset = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create metric for evaluation
    global_step = 0
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()


    # training loop
    for epoch in range(num_epoch):

        # clear metric at the start of each epoch
        train_metric.reset()

        model.train()

        for img, label in train_classification_dataset:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            preds = logits.argmax(dim=1)
            train_metric.add(preds, label)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            # log training loss to tensorboard
            logger.add_scalar("train/loss", loss.item(), global_step)

            global_step += 1

        # evaluate on validation set at the end of each epoch
        val_metric.reset()
        with torch.inference_mode():
            model.eval()

            for img, label in val_classification_dataset:
                img, label = img.to(device), label.to(device)

                logits = model(img)
                preds = logits.argmax(dim=1)
                val_metric.add(preds, label)

        train_acc = train_metric.compute()["accuracy"]
        val_acc = val_metric.compute()["accuracy"]

        logger.add_scalar("train/accuracy", train_acc, epoch)
        logger.add_scalar("val/accuracy", val_acc, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory for checkpointing
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Saved model checkpoint to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--exp_dir", type=str, default="logs", required=True, help="Directory to save logs and model checkpoints")
    parser.add_argument("--model_name", type=str, default="classifier", help="Model name for loading and saving")
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and validation")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")

    # pass all arguments to train_classification via kwargs
    args = parser.parse_args()
    train_classification(**vars(args))

