import lightning as L
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def preprocess():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)
    train_df.fillna(train_df.median(), inplace=True)
    test_df.fillna(test_df.median(), inplace=True)

    att_train = train_df.columns
    att_test = test_df.columns
    not_included = att_train ^ att_test
    Y_train = train_df["SalePrice"]
    train_df.drop(not_included, axis=1, inplace=True)
    X_train = train_df.drop("Id", axis=1)
    X_test = test_df.drop("Id", axis=1)
    id_test = test_df["Id"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    price_scaler = MinMaxScaler()
    Y_train = price_scaler.fit_transform(Y_train.values.reshape(-1, 1))

    return X_train, Y_train, X_test, id_test, price_scaler


# %%
class HouseModel(L.LightningModule):
    def __init__(self, price_scaler):
        super().__init__()

        self.example_input_array = torch.Tensor(5, 270)

        self.net = nn.Sequential(
            nn.Linear(270, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.criterion = nn.HuberLoss()
        self.scaler = price_scaler

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
        abs_error = self.scaler.inverse_transform(
            abs(y_pred.detach().numpy() - y.detach().numpy())
        ).mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("abs_error", abs_error, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        y_pred = self.net(x)
        price = self.scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))
        return price

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=1e-3)


def train(X_train, Y_train, X_test, id_test, price_scaler):
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    X_test = torch.FloatTensor(X_test)

    train_ds = TensorDataset(X_train, Y_train)
    train_dataloader = DataLoader(dataset=train_ds, shuffle=True, batch_size=32)
    model = HouseModel(price_scaler)
    summary = ModelSummary(max_depth=-1)
    early_stopping = EarlyStopping(monitor="train_loss", mode="min", patience=10)
    tb_logger = TensorBoardLogger("results", name="my_model")

    trainer = L.Trainer(
        max_epochs=100, callbacks=[early_stopping, summary], logger=tb_logger, gpus=0
    )

    trainer.fit(model=model, train_dataloaders=train_dataloader)

    # Inference on test set
    results = trainer.predict(model, X_test)
    results = [price[0][0] for price in results]
    results = pd.DataFrame({"Id": id_test, "SalePrice": results})
    results.to_csv("results.csv", index=False)


if __name__ == "__main__":
    X_train, Y_train, X_test, id_test, price_scaler = preprocess()
    train(X_train, Y_train, X_test, id_test, price_scaler)
