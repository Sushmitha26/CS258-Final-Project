import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from rsaenv import RSAEnv


# ======================================================
# Utility: Moving Average for Plot Smoothening
# ======================================================
def moving_average(arr, window=10):
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(arr, np.ones(window)/window, mode="valid")


# ======================================================
# Callback to record metrics each episode
# ======================================================
class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self.ep_block_rates = []

    def _on_step(self) -> bool:
        # Reward logging
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if any(dones):
            for info in infos:
                if "episode" in info:
                    self.ep_rewards.append(info["episode"]["r"])
                if "blocking_rate" in info:
                    self.ep_block_rates.append(info["blocking_rate"])

        return True


# ======================================================
# Training Function
# ======================================================
def train_dqn(train_files, capacity, save_prefix, timesteps=200_000):
    print(f"\n===== TRAINING DQN (capacity={capacity}) =====\n")

    env = RSAEnv(
        request_files=train_files,
        link_capacity=capacity,
        shuffle_files=True
    )
    env = Monitor(env)

    callback = EpisodeLogger()

    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=5e-4,
        batch_size=64,
        buffer_size=110_000,
        learning_starts=1_000,
        gamma=0.99,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
    )

    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(f"{save_prefix}.zip")

    print(f"Model saved as {save_prefix}.zip")

    return callback.ep_rewards, callback.ep_block_rates


# ======================================================
# Evaluation Function (deterministic=True)
# ======================================================
def evaluate_model(model_file, eval_files, capacity, episodes=10):
    print(f"\n===== EVALUATING {model_file} =====\n")

    model = DQN.load(model_file)

    env = RSAEnv(
        request_files=eval_files,
        link_capacity=capacity,
        shuffle_files=False
    )

    blocking_rates = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

        blocking_rates.append(info["blocking_rate"])

    avg_B = float(np.mean(blocking_rates))
    print(f"[RESULT] Average Blocking Rate B = {avg_B:.4f}")

    return avg_B


# ======================================================
# Plotting Function
# ======================================================
def plot_curves(rewards, blocks, tag):
    os.makedirs("plots", exist_ok=True)

    # Learning curve
    plt.figure(figsize=(6,4))
    plt.plot(moving_average(rewards, 10))
    plt.title(f"Learning Curve ({tag})")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward (10-ep MA)")
    plt.grid(True)
    plt.savefig(f"plots/{tag}_learning_curve.png")
    plt.close()

    # Blocking curve
    plt.figure(figsize=(6,4))
    plt.plot(moving_average(blocks, 10))
    plt.title(f"Blocking Rate Curve ({tag})")
    plt.xlabel("Episode")
    plt.ylabel("Avg Blocking Rate B (10-ep MA)")
    plt.grid(True)
    plt.savefig(f"plots/{tag}_blocking_curve.png")
    plt.close()

    print(f"Saved plots → plots/{tag}_*.png")


# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":

    train_dir = "data/train"
    eval_dir  = "data/eval"

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    eval_files  = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir)]

    # ----------------------------------------------------
    # PART 1: CAPACITY = 20
    # ----------------------------------------------------
    r20, b20 = train_dqn(
        train_files=train_files,
        capacity=20,
        save_prefix="rsa_dqn_cap20",
        timesteps=400_000,
    )
    plot_curves(r20, b20, "cap20")

    eval_B20 = evaluate_model("rsa_dqn_cap20", eval_files, capacity=20)

    # ----------------------------------------------------
    # PART 2: CAPACITY = 10
    # ----------------------------------------------------
    r10, b10 = train_dqn(
        train_files=train_files,
        capacity=10,
        save_prefix="rsa_dqn_cap10",
        timesteps=400_000,
    )
    plot_curves(r10, b10, "cap10")

    eval_B10 = evaluate_model("rsa_dqn_cap10", eval_files, capacity=10)

    print("\n===== DONE =====\n")
