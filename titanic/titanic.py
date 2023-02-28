import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
from keras.utils import to_categorical
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy

PASSENGER_ID_FROM = 892


def preprocess():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Drop unnecessary columns
    atts = ["Name", "PassengerId", "Cabin", "Ticket"]
    for ds in [train, test]:
        for att in atts:
            del ds[att]

    # Fill missing values
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    X_train = train.drop(labels="Survived", axis=1)
    Y_train = train["Survived"]
    X_test = test
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    # Scale data
    mm = MinMaxScaler()
    X_train = mm.fit_transform(X_train)
    X_test = mm.transform(X_test)

    Y_train = to_categorical(Y_train)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_train, Y_train, X_test


class TitanicModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        x = self.net(x)
        return F.softmax(x, dim=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_decoded = torch.where(y == 1)[1]
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        y_pred_prob = F.softmax(y_pred, dim=1)
        self.train_acc(y_pred_prob, y_decoded)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train():
    X_train, Y_train, X_test = preprocess()
    train_ds = TensorDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    model = TitanicModel()
    early_stopping = EarlyStopping(monitor="train_loss", mode="min", patience=10)
    mlf_logger = TensorBoardLogger("results", name="my_model")

    trainer = L.Trainer(max_epochs=100, gpus=0, callbacks=[early_stopping], logger=mlf_logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader)

    # Inference on test set
    results = trainer.predict(model, X_test)
    results = [torch.argmax(r).item() for r in results]
    results = pd.DataFrame(
        {"PassengerId": [i + PASSENGER_ID_FROM for i in range(len(results))], "Survived": results}
    )
    results.to_csv("results.csv", index=False)


if __name__ == "__main__":
    train()
