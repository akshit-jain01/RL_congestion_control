from stable_baselines3 import PPO
from env import CongestionEnv


import matplotlib.pyplot as plt


env = CongestionEnv()
model = PPO.load("cc_rl_model_3", env=env)

obs, _ = env.reset()

total_throughput = 0
total_delay = 0
steps = 30

for _ in range(steps):
    action, _ = model.predict(obs)
    action = int(action)
    obs, reward, terminated, truncated, _ = env.step(action)

    throughput = obs[0] * 10
    delay = obs[1] * 50 + 100

    total_throughput += throughput
    total_delay += delay

    loss = getattr(env, "last_loss", 0)

    window = env.window_sizes[env.idx]

    action_map = {0: "↓ DECREASE", 1: "→ SAME", 2: "↑ INCREASE"}

    print(
        f"CWND: {window} KB | "
        f"Action: {action_map[action]} | "
        f"Throughput: {throughput:.2f} | "
        f"Delay: {delay:.2f} | "
        f"Loss: {loss}"
    )


history = env.history

if len(history) == 0:
    print("No data collected!")


windows = [h[0] for h in history]
throughputs = [h[1] for h in history]
delays = [h[2] for h in history]
losses = [h[3] for h in history]
rewards = [h[4] for h in history]

plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(windows)
plt.title("CWND (KB)")

plt.subplot(4,1,2)
plt.plot(throughputs)
plt.title("Throughput (Mbps)")

plt.subplot(4,1,3)
plt.plot(delays)
plt.title("Delay (ms)")

plt.subplot(4,1,4)
plt.plot(losses)
plt.title("Loss")

plt.tight_layout()
plt.show()

print("RL Avg Throughput:", total_throughput / steps)
print("RL Avg Delay:", total_delay / steps)
