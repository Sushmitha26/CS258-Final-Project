import os
import csv

from dqn_runner import train_dqn, evaluate_model, plot_curves


NUM_RUNS = 10
TIMESTEPS = 400_000


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":

    ensure_dir("multi_run_results")
    ensure_dir("models")

    train_dir = "data/train"
    eval_dir  = "data/eval"

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    eval_files  = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir)]

    summary_rows = []

    for run in range(1, NUM_RUNS + 1):
        print(f"\n======================")
        print(f"   RUN {run}/{NUM_RUNS}")
        print(f"======================\n")

        tag = f"run{run}"

        # -------------------------
        # CAPACITY = 20
        # -------------------------
        r20, b20 = train_dqn(
            train_files=train_files,
            capacity=20,
            save_prefix=f"rsa_dqn_cap20_{tag}",
            timesteps=TIMESTEPS
        )
        plot_curves(r20, b20, f"cap20_{tag}")

        eval_B20 = evaluate_model(
            f"models/rsa_dqn_cap20_{tag}.zip",
            eval_files,
            capacity=20,
            episodes=10
        )

        # -------------------------
        # CAPACITY = 10
        # -------------------------
        r10, b10 = train_dqn(
            train_files=train_files,
            capacity=10,
            save_prefix=f"rsa_dqn_cap10_{tag}",
            timesteps=TIMESTEPS
        )
        plot_curves(r10, b10, f"cap10_{tag}")

        eval_B10 = evaluate_model(
            f"models/rsa_dqn_cap10_{tag}.zip",
            eval_files,
            capacity=10,
            episodes=10
        )

        summary_rows.append([run, eval_B20, eval_B10])

        print(f"Run {run} summary: cap20={eval_B20:.4f}, cap10={eval_B10:.4f}")

    # =======================
    # Save summary CSV
    # =======================
    with open("multi_run_results/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Cap20_BlockingRate", "Cap10_BlockingRate"])
        writer.writerows(summary_rows)

    print("\nAll 10 runs are complete!")
    print("Summary saved to multi_run_results/summary.csv")
