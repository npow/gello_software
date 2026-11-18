"""Replay a recorded episode on the real YAM at several speedups and measure tracking.

Finds the largest time-speedup factor the arm can physically follow, to bound
speed augmentation of teleop data. Replays the recorded leader commands open-loop
(no GELLO), so clear the workspace and stand by the e-stop.

Safety:
  - Refuses to start if the arm is >0.6 rad from the episode's first pose.
  - Eases to the start pose and back to the original pose at <=0.3 rad/s.
  - Aborts a run (and eases home) if any arm joint lags its command by >ABORT_ERR.

Usage:
    python scripts/replay_speedup_test.py --episode data/red-lid-conveyor/episode_0009 \
        --factors 1.0 1.25 1.5 2.0
"""

import argparse
import time

import numpy as np

RATE_HZ = 100.0
EASE_SPEED = 0.3  # rad/s
ABORT_ERR = 0.5  # rad


def ease(robot, target: np.ndarray) -> None:
    cur = np.asarray(robot.get_joint_pos(), dtype=float)
    steps = max(1, int(np.abs(target - cur).max() / EASE_SPEED * RATE_HZ))
    for q in np.linspace(cur, target, steps + 1)[1:]:
        robot.command_joint_pos(q.copy())
        time.sleep(1.0 / RATE_HZ)
    time.sleep(0.5)


def run(robot, t: np.ndarray, cmds: np.ndarray, k: float) -> dict:
    """Play cmds (N,7) over time t/k; return per-joint tracking errors."""
    tt = (t - t[0]) / k
    dur = tt[-1]
    errs, t0 = [], time.monotonic()
    while True:
        now = time.monotonic() - t0
        if now > dur:
            break
        cmd = np.array([np.interp(now, tt, cmds[:, j]) for j in range(cmds.shape[1])])
        robot.command_joint_pos(cmd.copy())
        meas = np.asarray(robot.get_joint_pos(), dtype=float)
        e = np.abs(cmd[:6] - meas[:6])
        errs.append(e)
        if e.max() > ABORT_ERR:
            print(f"  ABORT at {now:.2f}s: joint {e.argmax() + 1} lag {e.max():.2f} rad")
            robot.command_joint_pos(meas.copy())  # hold where it is
            return {"aborted": True, "errs": np.array(errs)}
        time.sleep(max(0.0, 1.0 / RATE_HZ - (time.monotonic() - t0 - now)))
    time.sleep(0.5)
    return {"aborted": False, "errs": np.array(errs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="data/red-lid-conveyor/episode_0009")
    ap.add_argument("--factors", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--channel", default="can_left")
    args = ap.parse_args()

    z = np.load(f"{args.episode}/trajectory.npz")
    t, cmds = z["timestamp"].astype(float), z["leader_actions"].astype(float)

    from i2rt.robots.get_robot import get_yam_robot

    robot = get_yam_robot(channel=args.channel)
    home = None
    try:
        home = np.asarray(robot.get_joint_pos(), dtype=float)
        gap = np.abs(cmds[0, :6] - home[:6]).max()
        if gap > 0.6:
            print(f"Arm is {gap:.2f} rad from the episode start pose; refusing.")
            return
        results = {}
        for k in args.factors:
            print(f"\n=== {k:.2f}x ({(t[-1] - t[0]) / k:.1f}s) ===")
            ease(robot, cmds[0])
            r = run(robot, t, cmds, k)
            results[k] = r
            p95 = np.percentile(r["errs"], 95, axis=0)
            print(f"  p95 lag per joint (rad): {np.round(p95, 3).tolist()}")
            print(f"  max lag per joint (rad): {np.round(r['errs'].max(0), 3).tolist()}")
            ease(robot, cmds[0])
            if r["aborted"]:
                print("  Stopping: arm could not follow this speed.")
                break
        print("\n=== Summary: worst-joint p95 / max lag (rad) ===")
        for k, r in results.items():
            p95 = np.percentile(r["errs"], 95, axis=0).max()
            tag = "ABORTED" if r["aborted"] else ""
            print(f"  {k:.2f}x: {p95:.3f} / {r['errs'].max():.3f} {tag}")
    finally:
        if home is not None:
            ease(robot, home)
        robot.close()


if __name__ == "__main__":
    main()
