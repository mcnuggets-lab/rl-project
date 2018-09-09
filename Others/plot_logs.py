import pandas as pd

df = pd.read_csv("./logs/FrozenLake-v0_train.log", names=["rewards"])
df["rewards"].rolling(50).mean().plot()
