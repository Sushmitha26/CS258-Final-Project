import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

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
# TRAINING FUNCTION (WITH CHECKPOINTING)
# ======================================================
def train_dqn(train_files, capacity, save_prefix, timesteps=200_000):

    print(f"\n===== TRAINING DQN (capacity={capacity}) =====\n")

    # --- make sure folders exist ---
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/best_models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # --- Env ---
    env = RSAEnv(
        request_files=train_files,
        link_capacity=capacity,
        shuffle_files=True
    )
    env = Monitor(env)

    episode_logger = EpisodeLogger()

    # --- Checkpoint callback ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"results/checkpoints/{save_prefix}",
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # --- Best model callback ---
    eval_env = RSAEnv(
        request_files=train_files,
        link_capacity=capacity,
        shuffle_files=False
    )
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"results/best_models/{save_prefix}",
        log_path=f"results/best_models/{save_prefix}",
        eval_freq=20_000,
        deterministic=True,
        render=False,
    )

    callbacks = [episode_logger, checkpoint_callback, eval_callback]

    # --- DQN ---
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=200_000,
        learning_starts=1_000,
        gamma=0.99,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
    )

    # --- TRAIN ---
    model.learn(total_timesteps=timesteps, callback=callbacks)

    # Save final model
    model.save(f"results/models/{save_prefix}.zip")
    print(f"Final model saved to results/models/{save_prefix}.zip")

    return episode_logger.ep_rewards, episode_logger.ep_block_rates


# ======================================================
# EVALUATION FUNCTION
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
# PLOTTING FUNCTION
# ======================================================
def plot_curves(rewards, blocks, tag):
    os.makedirs("results/plots", exist_ok=True)

    # Learning curve
    plt.figure(figsize=(6,4))
    plt.plot(moving_average(rewards, 10))
    plt.title(f"Learning Curve ({tag})")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward (10-ep MA)")
    plt.grid(True)
    plt.savefig(f"results/plots/{tag}_learning_curve.png")
    plt.close()

    # Blocking curve
    plt.figure(figsize=(6,4))
    plt.plot(moving_average(blocks, 10))
    plt.title(f"Blocking Rate Curve ({tag})")
    plt.xlabel("Episode")
    plt.ylabel("Avg Blocking Rate B (10-ep MA)")
    plt.grid(True)
    plt.savefig(f"results/plots/{tag}_blocking_curve.png")
    plt.close()

    print(f"Saved plots → results/plots/{tag}_*.png")

