import argparse

import lightning as L
import pandas as pd
import torch
from keras.utils import to_categorical
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, MetricCollection, Precision, Recall


class SpaceshipData(L.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()

        TRAIN_PATH = "data/train.csv"
        TEST_PATH = "data/test.csv"

        self.batch_size = batch_size

        self.train_df = pd.read_csv(TRAIN_PATH)
        self.test_df = pd.read_csv(TEST_PATH)
        self.x_train, self.y_train, self.scaler, info = self.preprocess(self.train_df)
        self.x_test, _, _, _ = self.preprocess(self.test_df, scaler=self.scaler, info=info)

    @property
    def input_dim(self):
        return self.x_train.shape[1]

    @staticmethod
    def preprocess(raw_df, scaler=None, info=None):
        if info is None:
            df = raw_df.dropna(subset=["Cabin"])

        else:
            df = raw_df.copy()
            df["Cabin"] = df["Cabin"].fillna(f"{info['Deck']}/{info['Num']}/{info['Side']}")
            print(df)

        df["Deck"] = df["Cabin"].apply(lambda x: x.split("/")[0])
        df["Num"] = df["Cabin"].apply(lambda x: int(float(x.split("/")[1])))
        df["Side"] = df["Cabin"].apply(lambda x: x.split("/")[2])

        info = {
            "Deck": df["Deck"].mode()[0],
            "Side": df["Side"].mode()[0],
            "Num": df["Num"].median(),
        }

        df.drop(["PassengerId", "Cabin", "Name"], axis=1, inplace=True)
        df.fillna(df.median(), inplace=True)
        df = pd.get_dummies(df)

        try:
            x = df.drop("Transported", axis=1)
            y = to_categorical(df["Transported"])
            y = torch.tensor(y, dtype=torch.float32)

        except KeyError:
            x = df
            y = None

        finally:
            if scaler is None:
                scaler = MinMaxScaler()
            x = scaler.fit_transform(x.values)
            x = torch.tensor(x, dtype=torch.float32)

            return x, y, scaler, info

    def train_dataloader(self):
        ds = TensorDataset(self.x_train, self.y_train)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def predict_dataloader(self):
        ds = TensorDataset(self.x_test)
        return DataLoader(ds, batch_size=self.x_test.shape[0], shuffle=False, drop_last=False)


class SpaceshipModel(L.LightningModule):
    def __init__(self, input_dim: int, lr: float = 1e-3):
        super().__init__()

        self.lr = lr
        self.example_input_array = torch.Tensor(5, input_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

        self.loss_fn = nn.BCELoss()
        self.train_metrics = MetricCollection(
            [
                Accuracy(num_classes=2, task="multiclass"),
                Precision(num_classes=2, task="multiclass"),
                Recall(num_classes=2, task="multiclass"),
            ],
            prefix="train_",
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        self.train_metrics(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        self.log_dict(self.train_metrics, prog_bar=False, logger=True, on_epoch=True, on_step=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        y_pred = self.net(x)
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def write_submission_csv(predictions, df):
    transported = torch.argmax(predictions[0], dim=1)
    transported = torch.where(transported == 1, True, False).numpy()
    df["Transported"] = transported
    df = df[["PassengerId", "Transported"]]
    df.to_csv("submission.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-c", "--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    data = SpaceshipData(batch_size=args.batch_size)
    input_dim = data.input_dim
    model = SpaceshipModel(input_dim=input_dim, lr=args.lr)
    ealry_stopping = EarlyStopping(monitor="train_loss", mode="min", patience=10)
    summary = ModelSummary(max_depth=-1)
    log_dir = "logs" if not args.debug else "logs/debug"
    tb_logger = TensorBoardLogger("logs", name="spaceship")

    if args.debug:
        trainer = L.Trainer(
            max_epochs=1,
            gpus=0,
            callbacks=[ealry_stopping, summary],
            logger=tb_logger,
            fast_dev_run=True,
            profiler="simple",
        )

    elif args.ckpt_path is None:
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            gpus=0,
            callbacks=[ealry_stopping, summary],
            logger=tb_logger,
            profiler="simple",
        )

    else:
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            gpus=0,
            callbacks=[ealry_stopping, summary],
            logger=tb_logger,
            profiler="simple",
        )

    trainer.fit(model=model, datamodule=data, ckpt_path=args.ckpt_path)
    predictions = trainer.predict(model=model, datamodule=data)
    write_submission_csv(predictions, data.test_df)


if __name__ == "__main__":
    main()
