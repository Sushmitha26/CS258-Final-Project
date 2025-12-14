# Routing and Spectrum Allocation using Deep Reinforcement Learning  
**CS 258 Final Project – Group 12**

## Overview
This project solves the **Routing and Spectrum Allocation (RSA)** problem in optical communication networks using **Deep Reinforcement Learning (DQN)**. The objective is to **minimize the request blocking rate** by selecting an available route (from predefined candidate paths) and allocating wavelengths according to:

- wavelength continuity 
- link capacity constraints  
- no wavelength conflicts on a link  
- smallest index wavelength allocation  

We evaluate performance under two link capacity settings:

- **Part 1:** capacity = 20  
- **Part 2:** capacity = 10  

We further perform **systematic hyperparameter tuning** using **Optuna**, running **80 trials**, and then select the best hyperparameter combination based on the **lowest mean blocking rate**.

---

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required dependencies:

```bash
pip install -r gym-custom-env-routing/requirements.txt
```

---

## Repository Structure
```
project/
├── data/
│   ├── train/                 # Training request files (used only for training)
│   └── eval/                  # Evaluation request files (used only for evaluation)
│
├── results/
│   ├── models/                # Final trained DQN models (.zip)
│   ├── checkpoints/           # Periodic checkpoints during training
│   ├── best_models/           # Best models saved during training via EvalCallback
│   ├── plots/                 # Learning/blocking/eval plots
│   ├── optuna/                # Optuna tuning outputs (CSV logs)
│   └── multi_run/             # Summary CSV for 10-run experiment
│
├── rsaenv.py                  # Custom RSA Gymnasium environment
├── nwutil.py                  # Network utilities + LinkState + topology generator
├── dqn_runner.py              # Final training + evaluation + plot generation
├── optuna_tune.py             # Optuna tuning script (cap=10)
└── multi_run_runner.py        # 10-run script + summary.csv
```

---

## Environment Summary (RSAEnv)

### Observation (state)
A dictionary with:
- `link_utilizations`: utilization of each link (fraction of occupied wavelengths)
- `source`: request source node
- `destination`: request destination node
- `holding_time`: request holding time

### Action
Discrete action space of size **2**:
- action chooses one of the two candidate paths for the source, destination

### Reward
- **+1** in case request is being allocated successfully
- **-1** in case of request being blocked

### Time slot dynamics
At each step:
1. existing lightpaths age by one time slot and expired wavelengths are freed  
2. a new request arrives  
3. agent selects a candidate path  
4. smallest index wavelength that is free on **all links of the path** is allocated, otherwise the request is blocked  

---

## How to Run

> Run commands from inside the `project/` directory.

### 1) Hyperparameter tuning (Optuna) — run first

This performs **80 Optuna trials** on **capacity = 10**. Each configuration is evaluated over multiple runs and the **mean blocking rate** is used as the parameter to select the best trial configuration.

```bash
python optuna_tune.py
```

**Outputs:**
- Optuna trial results CSV:
  ```
  results/optuna/optuna_results.csv
  ```
- Console logs show each trial’s hyperparameters and mean ± std blocking rate

After tuning finishes, the **best hyperparameter combination** (lowest mean blocking rate) is selected and then further taken by `dqn_runner.py`.

---

### 2) Multi-run final experiment (10 runs)

This script further runs **10 training runs** using the tuned hyperparameters and evaluates performance for both capacities. (10 and 20)

```bash
python multi_run_runner.py
```

**Outputs:**
- Trained models:
  ```
  results/models/
  ```
- Plots:
  ```
  results/plots/
  ```
- Summary CSV with blocking rates across runs:
  ```
  results/multi_run/summary.csv
  ```

---

## Plot Generation

For **each capacity (20 and 10)**, the following **three plots** are generated (six total):

1. **Learning Curve (training)**  
   - Averaged episode rewards (10-episode moving average) vs episode

2. **Blocking Rate Curve (training)**  
   - Averaged blocking rate **B** (10-episode moving average) vs episode

3. **Blocking Rate Curve (evaluation)**  
   - Blocking rate **B** vs evaluation episode  
   - Evaluation uses `predict(..., deterministic=True)` on `data/eval`

All plots are saved to:
```
results/plots/
```

---

## Evaluation Metric

Blocking rate per episode:

$$
B = \frac{1}{T} \sum_{t=0}^{T-1} b_t
$$

where:
- \(b_t = 1\) if request \(t\) is blocked, else 0  
- \(T = 100\) requests per episode  

Blocking rates are reported as **fractions** (e.g., 0.03 = 3%).

---

## Notes on Data Usage
- Training uses files only from `data/train/`
- Evaluation uses files only from `data/eval/`

---

## Where Results Are Saved

- **Optuna tuning results:** `results/optuna/optuna_results.csv`
- **10-run summary:** `results/multi_run/summary.csv`
- **Trained models:** `results/models/`
- **Checkpoints:** `results/checkpoints/`
- **Best models:** `results/best_models/`
- **Plots:** `results/plots/`

---

## References
- A. Asiri and B. Wang, *Deep Reinforcement Learning for QoT-Aware Routing, Modulation, and Spectrum Assignment in Elastic Optical Networks*, Journal of Lightwave Technology, 2025.  
- Stable-Baselines3 documentation  
- Optuna documentation


