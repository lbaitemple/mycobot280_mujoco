"""
Train PPO or SAC on the minimum-servo reaching task, then compare the learned
policy with the exact minimum-servo solver on the same targets.

    python rl/train.py --algo sac --steps 200000
    python rl/train.py --algo ppo --steps 1000000 --envs 8
    python rl/train.py --eval-only runs/sac_min_servo.zip --algo sac

Tuning: if the policy uses more servos than the solver, raise --servo-penalty;
if it stops reaching targets, lower it.
"""
import argparse, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rl"))
from min_servo_env import MinServoReachEnv          # noqa: E402
from min_servo_ik import Arm, min_servo_ik          # noqa: E402


def evaluate(model, env_kwargs, n=50, seed=123):
    env = MinServoReachEnv(**env_kwargs)
    arm = Arm(env_kwargs.get("xml_path", ROOT / "models" / "mycobot_280_m5.xml"))
    succ, used, opt, dists = 0, [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        q0, target = env.q0.copy(), env.target.copy()
        done = trunc = False
        while not (done or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(a)
        succ += info["is_success"]; used.append(info["servos_used"]); dists.append(info["distance"])
        res = min_servo_ik(arm, target, q0, tol=env.success_tol, restarts=6)
        opt.append(res.k if res.k is not None else np.nan)
    print(f"\nEvaluation on {n} targets")
    print(f"  success rate           : {succ/n:.0%}")
    print(f"  final distance (median): {np.median(dists)*1000:.1f} mm")
    print(f"  servos used by policy  : {np.mean(used):.2f}")
    print(f"  exact minimum (solver) : {np.nanmean(opt):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--envs", type=int, default=1, help="parallel envs (useful for PPO)")
    ap.add_argument("--servo-penalty", type=float, default=0.5)
    ap.add_argument("--model", default=str(ROOT / "models" / "mycobot_280_m5.xml"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--eval-only", default=None, help="path to a saved .zip policy")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env

    env_kwargs = dict(xml_path=a.model, servo_penalty=a.servo_penalty)
    Algo = PPO if a.algo == "ppo" else SAC

    if a.eval_only:
        model = Algo.load(a.eval_only)
    else:
        venv = make_vec_env(MinServoReachEnv, n_envs=a.envs, seed=a.seed, env_kwargs=env_kwargs)
        if a.algo == "ppo":
            model = PPO("MlpPolicy", venv, n_steps=max(2048 // a.envs, 64),
                        batch_size=256, gamma=0.98, learning_rate=3e-4, ent_coef=0.0,
                        seed=a.seed, verbose=1)
        else:
            # Lower target entropy than the default (-6) keeps SAC from wiggling every joint
            model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=1_000_000,
                        batch_size=256, gamma=0.98, target_entropy=-9.0, seed=a.seed, verbose=1)
        t = time.time()
        model.learn(total_timesteps=a.steps)
        out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
        path = out / f"{a.algo}_min_servo"
        model.save(path)
        print(f"Saved {path}.zip  ({(time.time()-t)/60:.1f} min)")
    evaluate(model, env_kwargs)


if __name__ == "__main__":
    main()
