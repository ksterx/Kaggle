import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split

raw_train_df = pd.read_csv("data/train.csv")
raw_test_df = pd.read_csv("data/test.csv")

# Drop unnecessary columns
train_df = raw_train_df.drop(["id"], axis=1)
test_df = raw_test_df.drop(["id"], axis=1)

# Split into x and y
x_train = train_df.drop(["Class"], axis=1)
y_train = train_df.pop("Class")

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train
)

automl = AutoML()
automl.fit(
    x_train,
    y_train,
    task="classification",
    eval_method="cv",
    n_splits=5,
    metric="accuracy",
    verbose=1,
)

a = automl.predict(x_val)
