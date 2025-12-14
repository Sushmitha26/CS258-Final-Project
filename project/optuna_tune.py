import os
import csv
import numpy as np
import optuna
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from rsaenv import RSAEnv

# ======================================================
# OPTUNA LOGGING
# ======================================================
optuna.logging.set_verbosity(optuna.logging.INFO)

# ======================================================
# GLOBAL CONFIG
# ======================================================
CAPACITY = 10
TIMESTEPS = 150_000          # fast enough for tuning
N_TRIALS = 80                # ~8 hrs total
N_RUNS_PER_TRIAL = 5

RESULT_DIR = "results/optuna"
os.makedirs(RESULT_DIR, exist_ok=True)

TRAIN_DIR = "data/train"
EVAL_DIR  = "data/eval"

TRAIN_FILES = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR)]
EVAL_FILES  = [os.path.join(EVAL_DIR, f) for f in os.listdir(EVAL_DIR)]

# ======================================================
# TRAIN + EVAL FUNCTION
# ======================================================
def train_and_eval(params, seed, tag):

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = RSAEnv(
        request_files=TRAIN_FILES,
        link_capacity=CAPACITY,
        shuffle_files=True
    )
    env = Monitor(env)

    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        buffer_size=params["buffer_size"],
        gamma=params["gamma"],
        exploration_fraction=params["exploration_fraction"],
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=1,
        target_update_interval=500,
        learning_starts=1_000,
        verbose=0,
    )

    model.learn(total_timesteps=TIMESTEPS)
    model.save(f"results/models/{tag}.zip")

    # ---------- Evaluation ----------
    eval_env = RSAEnv(
        request_files=EVAL_FILES,
        link_capacity=CAPACITY,
        shuffle_files=False
    )

    blocking_rates = []

    for _ in range(10):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = eval_env.step(action)
        blocking_rates.append(info["blocking_rate"])

    return float(np.mean(blocking_rates))


# ======================================================
# OPTUNA OBJECTIVE
# ======================================================
def objective(trial):

    # 👇 INCLUDE YOUR WORKING CONFIG AS CENTER
    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 3e-4, 1e-3, log=True
        ),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [100_000, 200_000]
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [64]
        ),
        "gamma": trial.suggest_float(
            "gamma", 0.98, 0.995
        ),
        "exploration_fraction": trial.suggest_float(
            "exploration_fraction", 0.25, 0.35
        ),
    }

    print("\n===================================")
    print(f"TRIAL {trial.number}")
    print(params)
    print("===================================")

    results = []

    for run in range(N_RUNS_PER_TRIAL):
        seed = trial.number * 100 + run
        tag = f"optuna_t{trial.number}_r{run}"

        avg_B = train_and_eval(params, seed, tag)
        results.append(avg_B)

    mean_B = float(np.mean(results))
    std_B  = float(np.std(results))

    print(f"Mean blocking rate = {mean_B:.4f} ± {std_B:.4f}")

    # ---------- Save CSV ----------
    csv_path = os.path.join(RESULT_DIR, "optuna_results.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "trial", "mean_B", "std_B",
                "learning_rate", "buffer_size",
                "batch_size", "gamma",
                "exploration_fraction"
            ])
        writer.writerow([
            trial.number, mean_B, std_B,
            params["learning_rate"],
            params["buffer_size"],
            params["batch_size"],
            params["gamma"],
            params["exploration_fraction"]
        ])

    return mean_B


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        study_name="rsa_dqn_cap10"
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=========== BEST RESULT ===========")
    print(f"Best blocking rate: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
