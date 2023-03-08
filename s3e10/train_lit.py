import argparse

import lightning as L
import numpy as np
import pandas as pd
import torch
from keras.utils import to_categorical
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy


class S3E10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 4):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.raw_train_df = pd.read_csv("data/train.csv")
        self.raw_test_df = pd.read_csv("data/test.csv")

        # Drop unnecessary columns
        train_df = self.raw_train_df.drop(["id"], axis=1)
        test_df = self.raw_test_df.drop(["id"], axis=1)

        # Split into x and y
        x_train = train_df.drop(["Class"], axis=1)
        y_train = to_categorical(train_df["Class"])
        self.x_test = test_df
        self.x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=y_train,
        )

        self.train_ds = TensorDataset(
            torch.from_numpy(self.x_train.values).float(),
            torch.from_numpy(y_train).float(),
        )

        self.val_ds = TensorDataset(
            torch.from_numpy(x_val.values).float(),
            torch.from_numpy(y_val).float(),
        )

        self.test_ds = TensorDataset(
            torch.from_numpy(self.x_test.values).float(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=len(self.test_ds), shuffle=False, num_workers=self.num_workers
        )

    @property
    def input_dim(self):
        return self.x_train.shape[1]


class S3E10Model(L.LightningModule):
    def __init__(self, input_dim: int, learning_rate: float = 1e-3):
        super().__init__()

        self.example_input_array = torch.Tensor(5, input_dim)

        self.learning_rate = learning_rate

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(),
        )
        self.loss_fn = nn.BCELoss()
        self.train_acc = Accuracy(num_classes=2, task="multiclass")
        self.val_acc = Accuracy(num_classes=2, task="multiclass")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        self.train_acc(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        self.val_acc(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        y_pred = self.net(x)
        return y_pred[:, 1]  # Need strength of class 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def save_predictions(predictions):
    predictions = np.argmax(predictions, axis=1)
    predictions = pd.DataFrame(predictions, columns=["Class"])
    predictions.to_csv("data/predictions.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-w", "--num_workers", type=int, default=4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-c", "--ckpt_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    datamodule = S3E10DataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    model = S3E10Model(input_dim=datamodule.input_dim, learning_rate=args.learning_rate)

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=True)
    summary = ModelSummary(max_depth=-1)
    logger = TensorBoardLogger("logs", name="s3e10")

    # Train
    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stopping, summary],
        logger=logger,
        fast_dev_run=args.debug,
    )
    trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)

    # Predict
    predictions = trainer.predict(model, datamodule)
    df = datamodule.raw_test_df
    df["Class"] = predictions[0].numpy()
    df = df[["id", "Class"]]
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
