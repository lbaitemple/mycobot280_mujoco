"""
PPO vs SAC on Gymnasium's Pendulum-v1 (swing-up).

Trains both for the same number of steps, then compares
  - average return over 20 deterministic episodes (closer to 0 is better; > -200 ~ solved)
  - action jitter: mean |a_t - a_(t-1)|, lower = smoother torque
and saves a plot of one swing-up for each (pendulum_compare.png).

    python rl/pendulum_ppo_vs_sac.py --steps 20000
"""
import argparse, time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC


def rollout(model, seed):
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    angles, torques, ret = [], [], 0.0
    for _ in range(200):
        a, _ = model.predict(obs, deterministic=True)
        angles.append(np.degrees(np.arctan2(obs[1], obs[0])))
        torques.append(float(a[0]))
        obs, r, term, trunc, _ = env.step(a)
        ret += r
    return ret, np.array(angles), np.array(torques)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    algos = {
        "SAC": lambda env: SAC("MlpPolicy", env, learning_rate=1e-3, seed=a.seed, verbose=0),
        # rl-zoo style PPO settings for Pendulum (it needs many more steps than SAC)
        "PPO": lambda env: PPO("MlpPolicy", env, n_steps=1024, gamma=0.9, gae_lambda=0.95,
                               learning_rate=1e-3, clip_range=0.2, use_sde=True,
                               sde_sample_freq=4, seed=a.seed, verbose=0),
    }
    results = {}
    for name, make in algos.items():
        t = time.time()
        model = make(gym.make("Pendulum-v1"))
        model.learn(a.steps)
        rets, jit = [], []
        for s in range(20):
            ret, ang, tq = rollout(model, 1000 + s)
            rets.append(ret); jit.append(np.mean(np.abs(np.diff(tq))))
        results[name] = (np.mean(rets), np.std(rets), np.mean(jit), rollout(model, 1000))
        print(f"{name}: return {np.mean(rets):7.1f} +/- {np.std(rets):5.1f}   "
              f"jitter {np.mean(jit):.3f}   train time {time.time()-t:5.0f} s")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for name, (_, _, _, (ret, ang, tq)) in results.items():
            ax[0].plot(ang, label=f"{name} (return {ret:.0f})")
            ax[1].plot(tq, label=name)
        ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("angle from upright (deg)")
        ax[1].set_ylabel("torque"); ax[1].set_xlabel("step")
        ax[0].legend(); ax[0].set_title(f"Pendulum-v1 swing-up after {a.steps} training steps")
        fig.tight_layout(); fig.savefig("pendulum_compare.png", dpi=150)
        print("saved pendulum_compare.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
