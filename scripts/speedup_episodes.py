"""Make time-sped-up copies of recorded episodes (speed augmentation).

Each output episode replays the source k times faster at the same 30 Hz:
output frame i takes source time k*i. Joint data is linearly interpolated,
video frames use the nearest source frame. Belt and arm speed up together.

Keep k <= 1.5 for the YAM: scripts/replay_speedup_test.py showed joint 2
can't follow 2x (lag >0.5 rad), while 1.5x tracked within ~0.3 rad p95.

Usage:
    python scripts/speedup_episodes.py --src data/red-lid-conveyor \
        --dst data/red-lid-conveyor-speedup --factors 1.25 1.5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CAMS = ("left", "middle", "right")


def speedup_episode(src: Path, dst: Path, k: float, fps: float = 30.0) -> int:
    z = np.load(src / "trajectory.npz")
    n_src = len(z["timestamp"])
    n_out = int(np.floor((n_src - 1) / k)) + 1
    s = np.arange(n_out) * k  # fractional source index for each output frame
    idx = np.arange(n_src)

    def interp(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            return np.interp(s, idx, a)
        return np.stack([np.interp(s, idx, a[:, j]) for j in range(a.shape[1])], 1)

    follower = interp(z["follower_joints"])
    leader = interp(z["leader_actions"])
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst / "trajectory.npz",
        step=np.arange(n_out),
        timestamp=z["timestamp"][0] + np.arange(n_out) / fps,
        follower_joints=follower,
        leader_actions=leader,
        tracking_error=np.abs(follower - leader),
    )

    nearest = np.clip(np.round(s).astype(int), 0, n_src - 1)
    for cam in CAMS:
        path = src / f"{cam}.mp4"
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
        cap.release()
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(
            str(dst / f"{cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for i in nearest:
            out.write(frames[min(i, len(frames) - 1)])
        out.release()

    meta = json.loads((src / "metadata.json").read_text())
    meta.update(num_frames=n_out, speedup=k, source_episode=src.name)
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    return n_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/red-lid-conveyor")
    ap.add_argument("--dst", default="data/red-lid-conveyor-speedup")
    ap.add_argument("--factors", type=float, nargs="+", default=[1.25, 1.5])
    args = ap.parse_args()

    src_eps = sorted(Path(args.src).glob("episode_*"))
    dst_root = Path(args.dst)
    out_idx = 0
    for k in args.factors:
        for ep in src_eps:
            n = speedup_episode(ep, dst_root / f"episode_{out_idx:04d}", k)
            print(f"{ep.name} x{k} -> episode_{out_idx:04d} ({n} frames)")
            out_idx += 1


if __name__ == "__main__":
    main()
