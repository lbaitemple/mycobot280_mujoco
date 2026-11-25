"""
Minimum-servo inverse kinematics for the myCobot 280 in MuJoCo.

Problem
-------
    minimise   ||q - q0||_0                (number of servos that move)
    subject to tip(q) = p_target           (forward kinematics of the MuJoCo model)
               q_lo <= q <= q_hi           (joint limits)

||.||_0 is combinatorial, but with 6 joints there are only 63 non-empty
subsets, so we search them exactly in order of size k = 1, 2, 3, ...
For each subset S the other joints stay locked at q0 and we solve the
small bounded least-squares problem  min_x ||tip(q0 with q_S = x) - p||^2
from several starting points. The first k that reaches the target within
`tol` is the minimum. Among all solutions of that size we return the one
with the smallest total motion (tie-break), and also list the others.

Note: feasibility of each subset is checked numerically (multi-start),
so "unreachable with k servos" is a strong numerical result, not a proof.
"""
import itertools
from dataclasses import dataclass, field
import numpy as np
import mujoco
from scipy.optimize import least_squares

JOINTS = [f"joint{i}" for i in range(1, 7)]


class Arm:
    """Thin kinematics wrapper around the MuJoCo model."""

    def __init__(self, xml_path, site="tip"):
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site(site).id
        ids = [self.m.joint(n).id for n in JOINTS]
        self.qadr = self.m.jnt_qposadr[ids]
        self.dadr = self.m.jnt_dofadr[ids]
        self.lo, self.hi = self.m.jnt_range[ids].T.copy()
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._jac = np.zeros((3, self.m.nv))
        # Joints that can never move the tip (e.g. J6 spins about the tip's own axis)
        # and the farthest the tip can get from the shoulder, from random sampling.
        rng = np.random.default_rng(0)
        samples = rng.uniform(self.lo, self.hi, size=(3000, 6))
        cols = np.zeros(6)
        shoulder = None
        reach = 0.0
        for q in samples:
            J = self.tip_jacobian(q)
            cols = np.maximum(cols, np.linalg.norm(J, axis=0))
            if shoulder is None:
                shoulder = self.d.xpos[self.m.body(self.m.jnt_bodyid[ids[1]]).id].copy()
            reach = max(reach, np.linalg.norm(self.d.site_xpos[self.site] - shoulder))
        self.position_joints = [j for j in range(6) if cols[j] > 1e-8]
        self.shoulder, self.max_reach = shoulder, reach * 1.01

    def set_q(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def tip(self, q):
        self.set_q(q)
        return self.d.site_xpos[self.site].copy()

    def tip_jacobian(self, q):
        self.set_q(q)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self._jac, None, self.site)
        return self._jac[:, self.dadr].copy()

    def hits_floor(self, q):
        """True if any moving link penetrates the floor at configuration q."""
        self.set_q(q)
        mujoco.mj_forward(self.m, self.d)
        for c in self.d.contact[: self.d.ncon]:
            if self.floor in (c.geom1, c.geom2) and c.dist < -1e-4:
                return True
        return False


@dataclass
class Solution:
    joints: list            # 0-based indices of servos that move
    q: np.ndarray           # full joint vector (rad)
    error: float            # tip position error (m)
    motion: float           # sum |dq| over moved joints (rad)

    @property
    def servos(self):
        return [j + 1 for j in self.joints]


@dataclass
class Result:
    k: int                                    # minimum number of servos (None if unreachable)
    best: Solution = None
    alternatives: list = field(default_factory=list)   # other subsets of the same size
    closest: Solution = None                  # best effort if unreachable


def _solve_subset(arm, S, q0, target, rng, restarts):
    S = list(S)
    lo, hi = arm.lo[S], arm.hi[S]

    def res(x):
        q = q0.copy(); q[S] = x
        return arm.tip(q) - target

    def jac(x):
        q = q0.copy(); q[S] = x
        return arm.tip_jacobian(q)[:, S]

    found = []
    starts = [np.clip(q0[S], lo, hi)] + [rng.uniform(lo, hi) for _ in range(restarts)]
    for x0 in starts:
        r = least_squares(res, x0, jac=jac, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        q = q0.copy(); q[S] = r.x
        found.append((np.linalg.norm(r.fun), q))
    return found


def min_servo_ik(arm, target, q0=None, tol=1e-3, restarts=8, avoid_floor=True,
                 max_k=6, seed=0, motion="sum"):
    """Return the smallest set of servos that puts the tip on `target`.

    target : (3,) metres, world frame (base of the robot at the origin)
    q0     : (6,) current joint angles in radians (default: home = zeros)
    tol    : position tolerance in metres (default 1 mm)
    motion : tie-break between equally small sets: "sum" (total rotation) or "max"
    """
    target = np.asarray(target, float)
    q0 = np.zeros(6) if q0 is None else np.asarray(q0, float)
    rng = np.random.default_rng(seed)
    measure = (lambda dq: np.abs(dq).sum()) if motion == "sum" else (lambda dq: np.abs(dq).max())

    if np.linalg.norm(arm.tip(q0) - target) < tol:
        return Result(k=0, best=Solution([], q0.copy(), np.linalg.norm(arm.tip(q0) - target), 0.0))

    if np.linalg.norm(target - arm.shoulder) > arm.max_reach:
        gap = np.linalg.norm(target - arm.shoulder) - arm.max_reach / 1.01
        far = Solution([], q0.copy(), gap, 0.0)
        return Result(k=None, closest=far)

    closest = None
    usable = arm.position_joints               # J6 is skipped: it can't move the tip
    for k in range(1, min(max_k, len(usable)) + 1):
        sols = []
        for S in itertools.combinations(usable, k):
            best_S = None
            for err, q in _solve_subset(arm, S, q0, target, rng, restarts):
                if closest is None or err < closest.error:
                    closest = Solution(list(S), q, err, measure(q - q0))
                if err > tol or (avoid_floor and arm.hits_floor(q)):
                    continue
                # every joint in S must actually move, otherwise a smaller set works
                if np.any(np.abs(q[list(S)] - q0[list(S)]) < 1e-4):
                    continue
                cand = Solution(list(S), q, err, measure(q - q0))
                if best_S is None or cand.motion < best_S.motion:
                    best_S = cand
            if best_S is not None:
                sols.append(best_S)
        if sols:
            sols.sort(key=lambda s: s.motion)
            return Result(k=k, best=sols[0], alternatives=sols[1:])
    return Result(k=None, closest=closest)


def describe(result, q0):
    if result.k is None:
        c = result.closest
        if not c.joints:
            return (f"Target not reachable: it is about {c.error*1000:.0f} mm beyond the arm's "
                    f"maximum reach.")
        if c.error < 1e-3:
            return ("Target can only be reached by driving part of the arm into the table "
                    "(floor is 30 mm below the base frame). Use --allow-floor to ignore this.")
        return (f"Target not reachable within the joint limits. Closest the tip gets is "
                f"{c.error*1000:.1f} mm away (moving servos {c.servos}).")
    if result.k == 0:
        return "The tip is already on the target. No servo needs to move."
    b = result.best
    lines = [f"Minimum servos to move: {result.k}  ->  servo(s) {b.servos}",
             f"Tip error: {b.error*1000:.3f} mm   total rotation: {np.degrees(b.motion):.1f} deg",
             "Servo   start(deg)   end(deg)   change(deg)"]
    for j in range(6):
        mark = "  <- moves" if j in b.joints else ""
        lines.append(f"  J{j+1}   {np.degrees(q0[j]):9.2f}  {np.degrees(b.q[j]):9.2f}  "
                     f"{np.degrees(b.q[j]-q0[j]):10.2f}{mark}")
    if result.alternatives:
        alts = ", ".join(f"{a.servos} ({np.degrees(a.motion):.0f} deg)" for a in result.alternatives)
        lines.append(f"Other {result.k}-servo options: {alts}")
    return "\n".join(lines)
