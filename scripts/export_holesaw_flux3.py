"""Export holesaw recordings to a 30 Hz joint-space LeRobot v3 dataset for FLUX 3 Action.

Segments and review edits come from export_holesaw_praxis.plan, so this dataset holds the
same segments as the reviewed 15 Hz Praxis export. State = follower joints (6 + gripper),
action = leader commands (6 + gripper), absolute, recorder convention (gripper 1 = open).

Run with a LeRobot v3 environment that also has mujoco and scipy (see .venv-lerobot3):
    .venv-lerobot3/bin/python scripts/export_holesaw_flux3.py \
        data/holesaw-conveyor data/lerobot/holesaw-conveyor-flux3-yam-30hz \
        --repo-id orchestrallabs/holesaw-conveyor-flux3-yam-30hz
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from export_holesaw_praxis import REVIEWED_CURATION, TASK, nearest, plan

FPS = 30
CAMERAS = ("middle", "left", "right")
JOINTS = [f"joint{i}" for i in range(1, 7)] + ["gripper"]


def resample(source, start, stop):
    """Interpolate joints onto an exact 30 Hz grid and pick the nearest source frame for video.

    start and stop are rows of the original recording, as plan reports them; a source trimmed
    in place records its offset in metadata.json.
    """
    curation = json.loads((source / "metadata.json").read_text()).get("curation", {})
    start, stop = (row - curation.get("original_start_row", 0) for row in (start, stop))
    with np.load(source / "trajectory.npz") as z:
        if not 0 <= start < stop <= len(z["timestamp"]):
            raise ValueError(f"Rows {start}:{stop} fall outside {source}")
        time = z["timestamp"][start:stop] - z["timestamp"][start]
        state, action = z["follower_joints"][start:stop], z["leader_actions"][start:stop]
    grid = np.arange(int(time[-1] * FPS) + 1) / FPS
    columns = [np.column_stack([np.interp(grid, time, values[:, j]) for j in range(7)]).astype("float32")
               for values in (state, action)]
    return columns[0], columns[1], nearest(time, grid) + start


def read_frames(source, rows):
    """Yield one RGB frame per requested source row; rows never decrease."""
    captures = [cv2.VideoCapture(str(source / f"{camera}.mp4")) for camera in CAMERAS]
    try:
        position, frames = -1, None
        for row in rows:
            while position < row:
                frames = [capture.read() for capture in captures]
                position += 1
                if not all(ok and frame.shape == (480, 640, 3) for ok, frame in frames):
                    raise ValueError(f"Invalid source frame {position} in {source}")
            yield [frame[:, :, ::-1] for _, frame in frames]
    finally:
        for capture in captures:
            capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--only", nargs="+", help="Source episode names, for a dry run")
    args = parser.parse_args()

    entries, skipped, _ = plan(sorted(args.source.glob("episode_*")), reviewed_curation=True)
    if args.only:
        entries = [entry for entry in entries if entry[0].name in args.only]
    if not entries:
        raise ValueError("No usable source episodes")

    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (7,), "names": JOINTS},
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    dataset = LeRobotDataset.create(
        args.repo_id, FPS, features, root=args.output, robot_type="i2rt_yam",
        streaming_encoding=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=20, preset="fast"))

    provenance, repeated, total = [], 0, 0
    for index, (source, origin, *_) in enumerate(entries):
        state, action, rows = resample(source, origin["source_start_row"], origin["source_stop_row_exclusive"])
        for i, images in enumerate(read_frames(source, rows)):
            frame = {"observation.state": state[i], "action": action[i], "task": TASK}
            frame.update({f"observation.images.{camera}": image for camera, image in zip(CAMERAS, images)})
            dataset.add_frame(frame)
        dataset.save_episode()
        repeated += int((np.diff(rows) == 0).sum())
        total += len(rows)
        provenance.append({"episode_index": index, **origin, "rows_at_30_hz": len(rows)})
        print(f"{index:03d} {source.name} ({len(rows)} rows)", flush=True)
    dataset.finalize()

    source_map = {
        "episodes": provenance,
        "discarded_short_segments": [
            {key: value for key, value in segment.items() if key != "rows_at_15_hz"} for segment in skipped],
        "reviewed_curation": REVIEWED_CURATION,
        "repeated_video_frames": repeated,
        "total_rows": total,
    }
    (args.output / "meta" / "source_episodes.json").write_text(json.dumps(source_map, indent=2) + "\n")
    print(f"{len(provenance)} episodes, {total} rows, {repeated} repeated video frames")


if __name__ == "__main__":
    main()
