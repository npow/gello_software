"""Export timestamped holesaw recordings to a LeRobot v2.1 Praxis YAM dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

FPS = 15
WIDTH, HEIGHT = 640, 400
CAMERAS = {"middle": "top_left", "left": "wrist_left", "right": "wrist_right"}
TASK = "Pick up the hole saws from the moving conveyor belt and place them in the red case"
PARKED = np.array([2.843328, 0.000954, 0.003624, -0.097086, -0.016594, -0.134852, 0.990736])
MODEL = (
    Path(__file__).resolve().parents[1]
    / "third_party/i2rt/i2rt/robot_models/arm/yam/v1/yam_linear_4310_d405.xml"
)
DATA = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
REVIEWED_CURATION = {
    "episode_0050": {"minimum_start_s": 15.5, "reason": "Trim the missed first grasp before the successful retry."},
    "episode_0078": {"exclude": True, "reason": "Multiple empty grasps while target objects continued down the belt."},
    "episode_0129": {"drop_first_segment": True, "reason": "Keep the segment after the 1.78 s source gap."},
}


def output_path(root, template, index, **kwargs):
    path = root / template.format(episode_chunk=index // 1000, episode_index=index, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def stats(values, axis=0, count=None):
    values = values.astype(np.float64, copy=False)
    return {
        "min": values.min(axis=axis).tolist(),
        "max": values.max(axis=axis).tolist(),
        "mean": values.mean(axis=axis).tolist(),
        "std": values.std(axis=axis).tolist(),
        "count": [len(values) if count is None else count],
    }


def camera_export(source, coords, path):
    cap = cv2.VideoCapture(str(source))
    sample_indices = set(np.linspace(0, len(coords) - 1, min(len(coords), 25), dtype=int))
    samples = []
    try:
        with subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-n", "-filter_threads", "1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
             "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-threads", "1",
             "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        ) as encoder:
            source_index = -1
            for output_index, target in enumerate(coords):
                while source_index < target:
                    frame_ok, frame = cap.read()
                    if not frame_ok or frame.shape != (480, 640, 3):
                        raise ValueError(f"Invalid source frame {source_index + 1} in {source}")
                    source_index += 1
                cropped = frame[40:440]
                encoder.stdin.write(cropped.tobytes())
                if output_index in sample_indices:
                    samples.append(cropped[::8, ::8, ::-1])
            encoder.stdin.close()
            if encoder.wait():
                raise ValueError(f"Encoding failed for {path}")
    finally:
        cap.release()
    rgb = np.array(samples).reshape(-1, 3, 1, 1) / 255.0
    return stats(rgb, count=len(samples))


def nearest(timestamps, grid):
    right = np.searchsorted(timestamps, grid).clip(1, len(timestamps) - 1)
    left = right - 1
    return np.where(grid - timestamps[left] <= timestamps[right] - grid, left, right)


def cartesian(model, joints, base_to_rig):
    data = mujoco.MjData(model)
    site = model.site("grasp_site").id
    addresses = [model.jnt_qposadr[model.joint(f"joint{i}").id] for i in range(1, 7)]
    xyz = np.empty((len(joints), 3))
    rotation = np.empty((len(joints), 3, 3))
    for i, row in enumerate(joints):
        data.qpos[addresses] = row[:6]
        mujoco.mj_kinematics(model, data)
        xyz[i] = data.site_xpos[site]
        rotation[i] = data.site_xmat[site].reshape(3, 3)
    xyz = xyz @ base_to_rig[:3, :3].T + base_to_rig[:3, 3]
    rotation = Rotation.from_matrix(base_to_rig[:3, :3] @ rotation)
    return np.column_stack((xyz, rotation.as_euler("xyz"), np.clip(1 - joints[:, 6], 0, 1)))


def as_arrow(values):
    values = np.ascontiguousarray(values)
    if values.ndim == 1:
        return pa.array(values)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), values.shape[1])


def plan(sources, reviewed_curation=False):
    entries, skipped = [], []
    row_offset = 0
    for source in sources:
        review = REVIEWED_CURATION.get(source.name, {}) if reviewed_curation else {}
        if review.get("exclude"):
            continue
        metadata_path = source / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        original_start = metadata.get("curation", {}).get("original_start_row", 0)
        if original_start:
            review = {}
        with np.load(source / "trajectory.npz") as z:
            time = z["timestamp"] - z["timestamp"][0]
            state, action = z["follower_joints"], z["leader_actions"]
        if (not np.isfinite(time).all() or not (np.diff(time) > 0).all()
                or state.shape != (len(time), 7) or action.shape != state.shape
                or not np.isfinite(state).all() or not np.isfinite(action).all()):
            raise ValueError(f"Invalid trajectory in {source}")
        boundaries = np.r_[0, np.flatnonzero(np.diff(time) > 0.100) + 1, len(time)]
        if review.get("drop_first_segment"):
            if len(boundaries) < 3:
                raise ValueError(f"Expected a recording gap in {source}")
            boundaries = boundaries[1:]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if "minimum_start_s" in review:
                start = max(start, np.searchsorted(time, review["minimum_start_s"]))
                if start >= stop:
                    continue
            segment = time[start:stop]
            frames = int((segment[-1] - segment[0]) * FPS) + 1
            provenance = {"source_episode": source.name, "source_start_row": int(start + original_start),
                          "source_stop_row_exclusive": int(stop + original_start)}
            if frames < 102:
                skipped.append(provenance | {"rows_at_15_hz": frames})
                continue
            grid = np.arange(frames) / FPS
            sampled = nearest(segment, grid + segment[0]) + start
            entries.append((source, provenance, grid, sampled, state[sampled], action[sampled], row_offset))
            row_offset += frames
    return entries, skipped, row_offset


def run_job(root, index, entry, model, transforms, right):
    source, provenance, grid, selected, state, action, row_offset = entry
    n = len(grid)
    right_pose = np.broadcast_to(right, (n, 7))
    columns = {
        "observation.state": np.column_stack((cartesian(model, state, transforms[0]), right_pose)).astype("float32"),
        "action": np.column_stack((cartesian(model, action, transforms[0]), right_pose)).astype("float32"),
        "observation.joint_state": state.astype("float32"),
        "action.joint": action.astype("float32"),
        "timestamp": grid.astype("float32"),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, index, dtype=np.int64),
        "index": np.arange(row_offset, row_offset + n, dtype=np.int64),
        "task_index": np.zeros(n, dtype=np.int64),
    }
    path = output_path(root, DATA, index)
    pq.write_table(pa.table({key: as_arrow(value) for key, value in columns.items()}), path)
    measurements = {key: stats(value if value.ndim == 2 else value[:, None]) for key, value in columns.items()}
    for source_camera, role in CAMERAS.items():
        key = f"observation.images.{role}"
        path = output_path(root, VIDEO, index, video_key=key)
        measurements[key] = camera_export(source / f"{source_camera}.mp4", selected, path)
    print(f"{index:03d} {source.name} ({n} rows)", flush=True)
    return (
        {"episode_index": index, "tasks": [TASK], "length": n},
        {"episode_index": index, "stats": measurements},
        {"episode_index": index, **provenance},
    )


def write_metadata(root, results, skipped, frames, transforms_file, reviewed_curation=False):
    episodes, measurements, provenance = zip(*results)
    meta = root / "meta"
    meta.mkdir()
    write_jsonl(meta / "episodes.jsonl", episodes)
    write_jsonl(meta / "episodes_stats.jsonl", measurements)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": TASK}])
    source_map = {"episodes": provenance, "discarded_short_segments": skipped}
    if reviewed_curation:
        source_map["reviewed_curation"] = REVIEWED_CURATION
    meta.joinpath("source_episodes.json").write_text(json.dumps(source_map, indent=2) + "\n")
    names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    pose_names = [f"{arm}_{name}" for arm in ("left", "right") for name in names]
    joint_names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
    features = {}
    for key in ("observation.state", "action", "observation.joint_state", "action.joint"):
        names = pose_names if key in ("observation.state", "action") else joint_names
        features[key] = {"dtype": "float32", "shape": [len(names)], "names": [names]}
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    for view in CAMERAS.values():
        features[f"observation.images.{view}"] = {
            "dtype": "video", "shape": [HEIGHT, WIDTH, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": HEIGHT, "video.width": WIDTH, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.fps": FPS, "video.channels": 3,
                     "video.is_depth_map": False, "has_audio": False},
        }
    info = {
        "codebase_version": "v2.1", "robot_type": "i2rt_yam", "fps": FPS, "chunks_size": 1000,
        "total_episodes": len(episodes), "total_frames": frames, "total_tasks": 1,
        "total_videos": 3 * len(episodes), "total_chunks": (len(episodes) + 999) // 1000,
        "splits": {"train": f"0:{len(episodes)}"}, "data_path": DATA, "video_path": VIDEO,
        "features": features,
    }
    meta.joinpath("info.json").write_text(json.dumps(info, indent=2) + "\n")
    calibration = (
        "A supplied base-to-rig transform maps each arm into the common rig frame."
        if transforms_file else
        "The Cartesian poses are expressed in each arm's own base frame. The source recordings "
        "have no measured common-frame extrinsics. Apply calibrated base-to-rig transforms in "
        "the training loader and use the same convention when serving the YAM checkpoint."
    )
    if transforms_file:
        meta.joinpath("base_to_rig.json").write_text(transforms_file.read_text())
    review_note = (
        "Three flagged recordings were reviewed in the overhead and left wrist views. "
        "Episode 50 starts at the retry after 15.5 s, episode 78 is excluded, and episode 129 "
        "keeps only its post-gap segment. The other recordings have not been audited for task success."
        if reviewed_curation else "Task success was not audited."
    )
    root.joinpath("README.md").write_text(f"""---
tags:
- robotics
- lerobot
- yam
---

# Holesaw conveyor at 15 Hz

{TASK}. The left YAM performs the task and the right YAM stays parked.

LeRobot v2.1 with **{len(episodes)} segments**, **{frames:,} synchronized rows**, and three H.264 streams.
The overhead stream is `top_left`, followed by `wrist_left` and `wrist_right`.
The videos are center-cropped from 640×480 to 640×400 for the YAM checkpoint's 16:10 inputs.

`observation.state` and `action` each store 14 absolute values, left arm followed by right.
Each arm stores metres, extrinsic XYZ Euler radians, and gripper (0 open, 1 closed).
The poses use MuJoCo forward kinematics at the i2rt YAM linear-gripper model's `grasp_site`.
{calibration}
The right state and target are reconstructed from the fixed parked joint vector in
`configs/yam_right_conveyor.yaml`. The left state comes from follower joints and the left
target from leader commands. The additional seven-value joint fields preserve the left
arm's source convention (gripper 1 open).

Timing uses nearest recorded rows on a 15 Hz grid anchored at each segment's first recorded timestamp.
All views share the selected source rows. Gaps over 100 ms split a recording, and
segments below the public trainer's 102-row minimum are recorded in
`meta/source_episodes.json` along with the source of each retained segment.
Group splits by source episode when holding out data.
{review_note}

This format is intended for a bimanual port of Runway's Praxis action-expert LoRA recipe.
That port must encode chunk-relative 20-D actions with the YAM checkpoint's normalization.
The public DROID LoRA trainer uses a single arm and two camera views.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-to-rig", type=Path, help="JSON array of two homogeneous 4x4 matrices, left then right")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--curate-reviewed", action="store_true", help="Apply the video review of recordings 50, 78, and 129")
    args = parser.parse_args()
    transforms = np.array(json.loads(args.base_to_rig.read_text())) if args.base_to_rig else np.stack([np.eye(4)] * 2)
    if (transforms.shape != (2, 4, 4) or not np.isfinite(transforms).all()
            or not np.allclose(transforms[:, 3], [0, 0, 0, 1])
            or not np.allclose(transforms[:, :3, :3] @ transforms[:, :3, :3].transpose(0, 2, 1), np.eye(3))
            or not np.allclose(np.linalg.det(transforms[:, :3, :3]), 1)):
        raise ValueError("Expected two proper homogeneous transforms")
    cv2.setNumThreads(1)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    right = cartesian(model, PARKED[None, :], transforms[1])[0]
    entries, skipped, frames = plan(sorted(args.source.glob("episode_*")), args.curate_reviewed)
    if not entries:
        raise ValueError("No usable source episodes")
    args.output.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_job, args.output, i, entry, model, transforms, right)
                for i, entry in enumerate(entries)]
        results = [job.result() for job in jobs]
    write_metadata(args.output, results, skipped, frames, args.base_to_rig, args.curate_reviewed)


if __name__ == "__main__":
    main()
