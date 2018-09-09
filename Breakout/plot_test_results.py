import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

key = "breakout_phyran_agent"

data = []
with open("./{}_result.txt".format(key), 'r') as f:
    for line in f:
        _, _, reward, step = line.split(":")
        reward = float(reward.split(",")[0].strip())
        step = int(step.strip())
        data.append((reward, step))

data = pd.DataFrame(data, columns=["reward", "step"])
print(data)

rewards_bin_width = 5
plt.hist(data["reward"], bins=np.arange(min(data["reward"]), max(data["reward"]) + rewards_bin_width, rewards_bin_width), label="score")
plt.legend()
plt.savefig("{}_score.png".format(key))
plt.close()


