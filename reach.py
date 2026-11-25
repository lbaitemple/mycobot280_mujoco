"""
Move the myCobot 280 tip to a 3D point while moving as few servos as possible.

Examples
--------
    python reach.py 0.15 -0.10 0.20                 # from the home pose
    python reach.py 0.15 -0.10 0.20 --start 0 -30 60 0 0 0   # start angles in degrees
    python reach.py 0.15 -0.10 0.20 --no-view       # just print the answer
    python reach.py 0.15 -0.10 0.20 --model models/mycobot_280_pi.xml

Coordinates are metres in the robot base frame (x forward, z up, origin at the
base). On macOS the 3D viewer needs `mjpython` instead of `python`.
"""
import argparse, pathlib, time
import numpy as np
from min_servo_ik import Arm, min_servo_ik, describe

HERE = pathlib.Path(__file__).resolve().parent


def animate(arm, q0, q1, target, seconds=2.5):
    import mujoco, mujoco.viewer
    m, d = arm.m, arm.d
    d.mocap_pos[0] = target
    d.qpos[arm.qadr] = q0
    d.ctrl[:] = q0
    mujoco.mj_forward(m, d)
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.9, -20, 135
        t0 = time.time()
        while v.is_running():
            s = min((time.time() - t0) / seconds, 1.0)
            s = 3 * s**2 - 2 * s**3                       # smooth start/stop
            q = q0 + s * (q1 - q0)                          # only the chosen servos change
            d.ctrl[:] = q                                   # drive the position servos
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(m.opt.timestep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x", type=float); ap.add_argument("y", type=float); ap.add_argument("z", type=float)
    ap.add_argument("--start", type=float, nargs=6, default=[0] * 6, metavar="DEG",
                    help="starting joint angles J1..J6 in degrees (default: home)")
    ap.add_argument("--model", default=str(HERE / "models" / "mycobot_280_m5.xml"))
    ap.add_argument("--tol", type=float, default=1.0, help="position tolerance in mm (default 1)")
    ap.add_argument("--tiebreak", choices=["sum", "max"], default="sum",
                    help="between equally small sets prefer least total rotation (sum) "
                         "or least single-servo rotation (max)")
    ap.add_argument("--allow-floor", action="store_true", help="don't reject poses that hit the floor")
    ap.add_argument("--no-view", action="store_true")
    a = ap.parse_args()

    arm = Arm(a.model)
    q0 = np.radians(a.start)
    target = np.array([a.x, a.y, a.z])
    t = time.time()
    res = min_servo_ik(arm, target, q0, tol=a.tol / 1000, motion=a.tiebreak,
                       avoid_floor=not a.allow_floor)
    print(f"Target {target} m   (solved in {time.time()-t:.2f} s)")
    print(describe(res, q0))
    if res.k:
        deg = np.round(np.degrees(res.best.q), 2).tolist()
        print("\nFor the real arm (pymycobot):  mc.send_angles(", deg, ", 30)")
    if res.k is not None and not a.no_view:
        try:
            animate(arm, q0, res.best.q, target)
        except RuntimeError as e:
            if "mjpython" not in str(e):
                raise
            import sys
            print("\nThe 3D viewer on macOS needs mjpython. Run the same command as:\n  mjpython "
                  + " ".join(sys.argv))


if __name__ == "__main__":
    main()
