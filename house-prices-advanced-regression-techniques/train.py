# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
