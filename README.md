# Routing and Spectrum Allocation using DRL (Deep Reinforcement Learning)
**CS 258 Final Project – TEAM 12**

## Overview
This project solves the **Routing and Spectrum Allocation (RSA)** problem in optical communication networks using **Deep Reinforcement Learning (DQN)**. The objective is to **minimize the request blocking rate** by selecting an available route (from predefined candidate paths) and allocating wavelengths according to:

- wavelength continuity along the path  
- link capacity constraints  
- no wavelength conflicts on a link  
- smallest-index wavelength allocation

We evaluate performance under two link-capacity settings:

- **Part 1:** capacity : 20  
- **Part 2:** capacity : 10  

We perform a **systematic hyperparameter tuning** using **Optuna**, running **80 trials**, and then select the best hyperparameter combination on the basis of the **lowest mean blocking rate**.

---

## Repository Structure for the project implementation
```
project/
├── data/
│   ├── train/                 # Training request files (used only for training)
│   └── eval/                  # Evaluation request files (used only for evaluation)
│
├── results/
│   ├── models/                # Final trained DQN models
│   ├── checkpoints/           # Periodic checkpoints during training
│   ├── best_models/           # Best models saved during training via EvalCallback
│   ├── plots/                 # Learning/blocking/eval plots
│   ├── optuna/                # Optuna tuning outputs (CSV logs)
│   └── multi_run/             # Summary CSV for 10 runs experiment
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
- `source`: request source node
- `destination`: request destination node
- `holding_time`: request holding time
- `link_utilizations`: utilization of each link 

### Action
Discrete action space of size **2**:
- action chooses one of the two candidate paths for the source and destination

### Reward
- **+1** in case request is allocated successfully
- **-1** in case request is blocked

### Time-slot dynamics
At each step:
1. existing lightpaths age by one time slot and expired wavelengths are freed  
2. a new request arrives  
3. agent selects a candidate path  
4. smallest-index wavelength that is free on **all links of the path** is allocated, otherwise the request is blocked  

---

## How to Run

> Run commands from inside the `project/` directory.


### 1) Hyperparameter tuning (Optuna) — **run first**
This runs **80 trials** on **capacity=10** and logs trial results.

```bash
python optuna_tune.py
```

**Outputs:**
- Optuna trial results CSV:
  - `results/optuna/optuna_results.csv`
- Console logs show every trials parameters + mean/std blocking rate
- The script prints the best hyperparameters at the end

**After Optuna finishes:**
- Copy the **best hyperparameters** (which you get from the Optuna output) into the `dqn_runner.py` so final experiments use the tuned settings.

---

### 2) Multi-run final experiment (10 runs) — **run after Optuna**
This runs **10 independent training runs** (both capacities) using the best hyperparameters we got from optuna and writes a summary CSV.

```bash
python multi_run_runner.py
```

**Outputs:**
- Model saved under(here in this directory, uploading models for our best run):
  - `results/models/`
- Plots saved under(here in this directory, uploading plots for our best run):
  - `results/plots/`
- Summary CSV with blocking rates for each run (you can verify the best run from here):
  - `results/multi_run/summary.csv`

`summary.csv` contains:
- per-run blocking rate for capacity 20  
- per-run blocking rate for capacity 10  

---

## Plot Generation (what gets produced)
For **each capacity** (20 and 10), we generate **3 plots** (6 total):

1. **Learning curve (training)**  
   - averaged episode rewards (10-episode moving average) vs episode

2. **Objective / blocking curve (training)**  
   - averaged blocking rate **B** (10-episode moving average) vs episode

3. **Objective / blocking curve (evaluation)**  
   - blocking rate **B** vs evaluation episode  
   - evaluated on `data/eval` using `predict(..., deterministic=True)`

All plots are saved to:
- `results/plots/`

---

## Evaluation Metric
Blocking rate per episode:

$$
B = \frac{1}{T}\sum_{t=0}^{T-1} b_t
$$

where:
- $b_t = 1$ if request $t$ is blocked, else 0  
- $T = 100$ requests per episode  

Please note that the Blocking rate values are reported as **fractions** (e.g., 0.03 = 3%).

---

## Results

Below are the **six required plots** (three per capacity setting). All plots are generated using a 10-episode moving average where applicable.

### Capacity = 10

#### Learning Curve (Training)
![Cap10 Learning Curve](results/plots/cap10_run10_learning_curve.png)  
*Shows a steady improvement in average episodic reward under tighter capacity constraints.*

#### Blocking Rate Curve (Training)
![Cap10 Blocking Curve](results/plots/cap10_run10_blocking_curve.png)  
*Exhibits a decreasing trend in blocking rate as the agent learns to take better routing decisions.*

#### Blocking Rate Curve (Evaluation)
![Cap10 Eval Blocking Curve](results/plots/cap10_run10_eval_blocking_curve.png)  
*Evaluation on unseen requests using deterministic policy*

### Capacity = 20

#### Learning Curve (Training)
![Cap20 Learning Curve](results/plots/cap20_run10_learning_curve.png)  
*Exhibits a swift convergence to near-optimal reward due to higher available capacity.*

#### Blocking Rate Curve (Training)
![Cap20 Blocking Curve](results/plots/cap20_run10_blocking_curve.png)  
*Blocking rate remains close to zero for most of the episodes, indicating enough resources.*

#### Blocking Rate Curve (Evaluation)
![Cap20 Eval Blocking Curve](results/plots/cap20_run10_eval_blocking_curve.png)  
*Almost near to zero blocking during evaluation, confirming strong generalization with higher capacity.*

---

## Training Setup and Hyperparameter Tuning
- Algorithm: **DQN - Stable-Baselines3**
- Separate agents trained for capacities 10 and 20
- **Optuna** was used for systematic hyperparameter tuning (80 trials)
- Each trial was evaluated over multiple runs to account for stochasticity
- Final configuration was then selected based on : **lowest mean blocking rate**

## Notes on Data Usage 
- Training uses files only from: `data/train/`  
- Evaluation uses files only from: `data/eval/`  
- The tuning script trains only on the `data/train/` and evaluates on `data/eval/`

---

## Where your results are being saved
- **Optuna tuning CSV:** `results/optuna/optuna_results.csv`
- **10-run summary CSV:** `results/multi_run/summary.csv`
- **Trained models:** `results/models/`
- **Checkpoints:** `results/checkpoints/`
- **Best models:** `results/best_models/`
- **Plots:** `results/plots/`

---

## References
- A. Asiri and B. Wang, “Deep Reinforcement Learning for QoT-Aware Routing, Modulation, and Spectrum Assignment in Elastic Optical Networks,” *JLT*, 2025.  
- Stable-Baselines3 documentation  
- Optuna documentation
