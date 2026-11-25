"""
Watch a trained PPO or SAC policy drive the myCobot 280 in the MuJoCo viewer.

Train first (no viewer needed, plain python is fine):
    python rl/train.py --algo ppo --steps 1000000 --envs 8      # -> runs/ppo_min_servo.zip

Then watch it (macOS: use mjpython for anything that opens the viewer):
    mjpython rl/play_policy.py runs/ppo_min_servo.zip                       # random targets
    mjpython rl/play_policy.py runs/ppo_min_servo.zip --target 0.15 -0.10 0.20
    python   rl/play_policy.py runs/ppo_min_servo.zip --no-view            # just print results

The red sphere is the target. Close the window or press Ctrl+C to stop.
"""
import argparse, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "rl"))
from min_servo_env import MinServoReachEnv   # noqa: E402


def load_policy(path, algo):
    from stable_baselines3 import PPO, SAC
    if algo == "auto":
        algo = "sac" if "sac" in pathlib.Path(path).name.lower() else "ppo"
    return (SAC if algo == "sac" else PPO).load(path, device="cpu")


def run_episode(env, model, target, q0, viewer=None, step_time=0.04):
    opts = {}
    if target is not None:
        opts["target"] = target
    if q0 is not None:
        opts["q0"] = q0
    obs, _ = env.reset(options=opts or None)
    start = env.q.copy()
    done = trunc = False
    info = {"distance": float(np.linalg.norm(env.target - env.d.site_xpos[env.site])),
            "servos_used": 0, "is_success": False}
    if viewer is not None:
        viewer.sync(); time.sleep(0.5)
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(action)
        if viewer is not None:
            if not viewer.is_running():
                return None
            viewer.sync()
            time.sleep(step_time)
    moved = [j + 1 for j in range(6) if abs(env.q[j] - start[j]) > env.eps]
    print(f"target {np.round(env.target, 3)}  ->  "
          f"{'REACHED' if info['is_success'] else 'missed'}  "
          f"error {info['distance']*1000:.1f} mm  steps {env.t}  servos moved {moved}")
    print("   end angles (deg):", np.round(np.degrees(env.q), 2).tolist())
    if viewer is not None:
        time.sleep(1.5)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", help="saved policy .zip from rl/train.py")
    ap.add_argument("--algo", choices=["auto", "ppo", "sac"], default="auto")
    ap.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="fixed target in metres (default: a new random target each episode)")
    ap.add_argument("--start", type=float, nargs=6, metavar="DEG",
                    help="start joint angles in degrees (default: home if --target is given, else random)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--model", default=str(ROOT / "models" / "mycobot_280_m5.xml"))
    ap.add_argument("--no-view", action="store_true")
    a = ap.parse_args()

    model = load_policy(a.policy, a.algo)
    fixed = a.target is not None
    env = MinServoReachEnv(xml_path=a.model, random_start=not fixed and a.start is None)
    env.reset(seed=0)
    q0 = np.radians(a.start) if a.start is not None else (np.zeros(6) if fixed else None)
    target = np.array(a.target) if fixed else None
    episodes = 1 if fixed else a.episodes

    if a.no_view:
        for _ in range(episodes):
            run_episode(env, model, target, q0)
        return

    import mujoco.viewer
    try:
        viewer = mujoco.viewer.launch_passive(env.m, env.d)
    except RuntimeError as e:
        if "mjpython" in str(e):
            sys.exit("The 3D viewer on macOS needs mjpython. Run:\n  mjpython " + " ".join(sys.argv))
        raise
    with viewer:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.9, -20, 135
        viewer.cam.lookat[:] = [0, 0, 0.17]
        for _ in range(episodes):
            if run_episode(env, model, target, q0, viewer) is None:
                break
        while viewer.is_running():       # keep the last pose on screen until closed
            viewer.sync(); time.sleep(0.05)


if __name__ == "__main__":
    main()
