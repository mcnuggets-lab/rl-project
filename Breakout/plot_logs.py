import json
import pandas as pd
from matplotlib import pyplot as plt

with open("./dqn_BreakoutDeterministic-v4_log.json", 'r') as f:
    json_string = f.read()
data = json.loads(json_string)
data = pd.DataFrame(data)
print(data)

# plt.plot(data["episode_reward"].rolling(500).mean())
# plt.savefig("breakout-phydqn-rewards.png")
# plt.plot(data["mean_q"].rolling(500).mean())
# plt.savefig("breakout-phydqn-meanq.png")


