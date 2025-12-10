import glob
from rsaenv import RSAEnv

# Adjust pattern if filenames are different (e.g., *.csv or *.txt)
train_files = sorted(glob.glob("data/train/*.csv"))  # or "*.txt"

env = RSAEnv(request_files=train_files, link_capacity=20)

obs, info = env.reset()
print("Initial obs keys:", obs.keys(), "info:", info)

done = False
step = 0
while not done:
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    done = terminated or truncated
    if step <= 5 or done:
        print(f"step {step}: reward={reward}, blocked={info['blocked']}, blocking_rate={info['blocking_rate']:.3f}")

print("Episode finished in", step, "steps")
