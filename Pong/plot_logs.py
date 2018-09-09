import pandas as pd
from matplotlib import pyplot as plt

env_name = "Pong-v0"
df = pd.read_csv("logs/{}_pg-test3.log".format(env_name))
print(df)

plt.plot(df["score"].rolling(500).mean())
plt.show()


