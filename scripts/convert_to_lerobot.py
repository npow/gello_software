"""Convert GELLO/YAM teleop episodes to a LeRobot (v2.x, openpi-pinned) dataset.

Each source episode dir holds trajectory.npz (follower_joints, leader_actions),
metadata.json and left/middle/right.mp4. State = follower joints (6 + gripper),
action = leader commands (6 + gripper), absolute, at the recorded 30 Hz.

Run with the openpi-pinned lerobot (see .venv-lerobot):
    .venv-lerobot/bin/python scripts/convert_to_lerobot.py \
        --src data/red-lid-conveyor --repo-id npow/red-lid-conveyor \
        --task "Pick up the red lid from the conveyor belt" --push --private
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

CAMS = ("left", "middle", "right")
MOTORS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def read_frames(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="episode root dir(s)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--task",
        default="Pick up the red lid from the conveyor belt",
        help="language instruction stored on every frame",
    )
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    out = LEROBOT_HOME / args.repo_id
    if out.exists():
        shutil.rmtree(out)

    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (7,), "names": [MOTORS]},
    }
    for cam in CAMS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type="i2rt_yam",
        features=features,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    eps = [e for s in args.src for e in sorted(Path(s).glob("episode_*"))]
    for i, ep in enumerate(eps):
        z = np.load(ep / "trajectory.npz")
        state = z["follower_joints"].astype(np.float32)
        action = z["leader_actions"].astype(np.float32)
        frames = {cam: read_frames(ep / f"{cam}.mp4") for cam in CAMS}
        n = min(len(state), *(len(f) for f in frames.values()))
        for t in range(n):
            frame = {"observation.state": state[t], "action": action[t], "task": args.task}
            for cam in CAMS:
                frame[f"observation.images.{cam}"] = frames[cam][t]
            ds.add_frame(frame)
        ds.save_episode()
        print(f"[{i + 1}/{len(eps)}] {ep} ({n} frames)", flush=True)

    if args.push:
        ds.push_to_hub(private=args.private)
        print(f"Pushed https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
