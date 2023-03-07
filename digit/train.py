import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from keras.utils import to_categorical
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision import transforms as T


class DigitDataset(Dataset):
    def __init__(self, x, y, transform):
        super().__init__()

        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.transform(self.x[index])
        if self.y is None:
            return x
        else:
            return x, self.y[index]

    def __len__(self):
        return len(self.x)


class DigitDataModule(L.LightningDataModule):
    def __init__(self, train_ratio=0.8):
        super().__init__()

        train_df = pd.read_csv("data/train.csv")
        self.test_df = pd.read_csv("data/test.csv")

        x_train = train_df.drop(labels=["label"], axis=1)
        y_train = train_df["label"]
        train_size = int(len(train_df) * train_ratio)
        x_train, x_val = x_train[:train_size], x_train[train_size:]
        y_train, y_val = y_train[:train_size], y_train[train_size:]
        train_array = np.array(x_train.values)
        mean = train_array.mean()
        std = train_array.std()

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((mean,), (std,)),
            ]
        )

        def to_tensorlist(df):
            return [np.reshape(i, (28, 28)).astype(np.float32) for i in df.values]

        self.x_train = to_tensorlist(x_train)
        self.x_val = to_tensorlist(x_val)
        self.x_test = to_tensorlist(self.test_df)
        self.y_train = torch.tensor(to_categorical(y_train))
        self.y_val = torch.tensor(to_categorical(y_val))

    def show_image(self, index):
        plt.imshow(self.x_train[index])
        plt.title(self.y_train[index])
        plt.show()

    def train_dataloader(self):
        return DataLoader(
            DigitDataset(self.x_train, self.y_train, self.transform),
            batch_size=32,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            DigitDataset(self.x_val, self.y_val, self.transform),
            batch_size=32,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )

    def predict_dataloader(self):
        return DataLoader(
            DigitDataset(self.x_test, None, self.transform),
            batch_size=len(self.test_df),
            shuffle=False,
            drop_last=False,
        )


class DigitModel(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.example_input_array = torch.zeros((1, 1, 28, 28))

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )

        self.loss_fn = nn.BCELoss()
        self.train_acc = Accuracy(num_classes=10, task="multiclass")
        self.val_acc = Accuracy(num_classes=10, task="multiclass")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        self.train_acc(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        self.val_acc(y_pred, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        y_pred = self.net(x)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-c", "--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    dm = DigitDataModule()
    model = DigitModel()
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    summary = ModelSummary(max_depth=-1)
    logger = TensorBoardLogger("logs", name="digit")
    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[early_stopping, summary],
        logger=logger,
        fast_dev_run=args.debug,
        profiler="simple",
        gpus=1,
    )
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)
    predictions = trainer.predict(model, dm)
    predictions = torch.argmax(predictions[0], dim=1)
    df = pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions})
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
