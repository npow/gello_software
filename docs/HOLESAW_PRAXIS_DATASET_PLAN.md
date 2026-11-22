# Holesaw conveyor dataset for Praxis-1: plan

Date: 2026-11-21. Status: plan only. Nothing has been converted or re-uploaded yet.

Target: `orchestrallabs/holesaw-conveyor-three-camera` on Hugging Face, used to post-train Runway's Praxis-1 from the YAM checkpoint.

## 1. What is on disk

Source: `data/holesaw-conveyor/`, episodes 48–153 (106 recordings, 144,910 frames, 1.41 hours).

- Each episode has `trajectory.npz` (7 follower joints, 7 leader commands, timestamps), `left.mp4`, `middle.mp4`, `right.mp4`, `metadata.json` and `recording.rrd`.
- Video is MPEG-4, 640x480 (4:3), labelled 30 fps.
- The real sample rate is 27.3–29.6 Hz (median 28.6 Hz), not 30.
- The old-rig episodes 9–47 are no longer in the folder.
- `dataset_manifest.json` is stale. It lists 66 episodes and stops at 83.

## 2. What the Praxis-1 YAM checkpoint expects

Target checkpoint: `runwayml/praxis-1-yam` (LeRobot build: `runwayml/praxis-1-yam-lerobot`).

These come from the model cards, Runway's repo (`praxis/bimanual.py`, `configs/serve_yam.yaml`, `tests/data/praxis-1-yam.json`) and the LeRobot Praxis page, read through a web-fetch agent. I have not run the code. Verify before training.

| Item | YAM checkpoint | Our data today |
| --- | --- | --- |
| Arms | Two (left first, then right) | Left arm only |
| Cameras | Three: `top_left` (overhead), `wrist_left`, `wrist_right` | Three: `middle` (overhead), `left` wrist, `right` wrist |
| Views | 16:10 (ZED 1920x1200), fitted to 256x416 tiles; the server "refuses views at another aspect" | 640x480 (4:3) |
| Rate | 15 Hz, 24-step chunks, 93-frame windows | ~28.6 Hz, labelled 30 |
| State | Per arm: end-effector pose `[xyz, quat_wxyz]` in metres in the rig frame, plus gripper. LeRobot page: `observation.state (14,)` as `[x, y, z, roll, pitch, yaw, gripper]` per arm | 7 joint values (6 joints + gripper) |
| Action | Per arm: Cartesian pose target plus gripper. LeRobot page: `action (14,)`, absolute target per arm; the model turns it into chunk-relative xyz + rot6d inside | 7 leader joint values |
| Gripper | 0 = open, 1 = closed | Opposite: 1 = open, 0 = closed (checked on episode 160: 0.95 with fingers apart, command 0 while holding a hole saw). Must be flipped |
| Task text | Verbatim to the text encoder, no closing period, e.g. "Pick up the pen from the center of the table, then place the pen in the bag" | Has a closing period |
| Episode length | At least 102 rows at 15 Hz (6.7 s) | All are 35–74 s |

The camera layout is a good match. Three things do not match: one arm instead of two, joints instead of Cartesian poses, and 4:3 instead of 16:10.

Fine-tuning from the YAM checkpoint: the published docs say there is no released recipe.
- LeRobot page: "forward (fine-tuning) is single-arm only, since the praxis repository has no bimanual training recipe yet."
- Runway training README: a rig with two arms "needs new code in `praxis/single_arm.py` and in the data module."
- Runway's write-up says its own YAM fine-tunes "follow the same recipe" as DROID, but that code is not in the repo.
- Nothing published says how to feed a single arm to the bimanual checkpoint (no padding or masking mode).

The sources disagree on details (14-D Euler vs 16-D quaternion state, absolute vs chunk-relative action). The dataset should store absolute poses; both readings can be derived from that.

Sources:
- https://huggingface.co/runwayml/praxis-1-yam
- https://huggingface.co/runwayml/praxis-1-yam-lerobot
- https://huggingface.co/docs/lerobot/main/en/praxis
- https://raw.githubusercontent.com/runwayml/praxis-1-oss/main/README.md
- https://raw.githubusercontent.com/runwayml/praxis-1-oss/main/praxis/train/README.md
- https://raw.githubusercontent.com/runwayml/praxis-1-oss/main/praxis/bimanual.py
- https://raw.githubusercontent.com/runwayml/praxis-1-oss/main/tests/data/praxis-1-yam.json
- https://runway.com/research/praxis-1-video-pretraining-to-robot-control

## 3. Problem with the current upload

The repo already holds 55 episodes (48–102) as LeRobot v2.1, AV1 video, labelled 30 fps.

- The values and frame counts match the recordings.
- The timing does not. The data was recorded at about 28.6 Hz and stamped as 30 fps, so every timestamp is about 5–9% too short.
- It stores joints only, with no end-effector pose, and uses camera keys `left`, `middle`, `right`.
- It is missing episodes 103–153.

It should be replaced, not added to.

## 4. Decisions needed

1. **Training route.** How will you post-train from the YAM checkpoint? The public repo has no bimanual training recipe. If you have a recipe or contact at Runway, its data spec should override this plan.
2. **Right arm (decided).** The right arm was parked for every episode; only the left arm moved, and all three cameras recorded. Use the constant right-arm joint vector `[2.843328, 0.000954, 0.003624, -0.097086, -0.016594, -0.134852, 0.990736]` (6 joints + gripper, recorder convention) for both state and action in the right-arm slots, and its forward-kinematics pose for the Cartesian columns.
3. **Poses from joints.** Recommended: compute the end-effector pose by forward kinematics from `i2rt/robot_models/arm/yam/yam.urdf`, and keep the raw joints as extra columns.
4. **Camera aspect.** Recommended: keep native 640x480 in the dataset and name the keys `top_left`, `wrist_left`, `wrist_right` (`middle`, `left`, `right`). Do the 16:10 crop (640x400) at training time, after checking the hole saws and case survive it.
5. **Replace the existing upload, and hold out about 5 episodes for evaluation.** Recommended: yes to both.

## 5. Data quality work

Already checked on all 106 episodes: no missing files, no NaNs, and the three videos have exactly one frame per trajectory row.

Flagged so far:

| Episode | Issue |
| --- | --- |
| 50 | Gripper still closed at the end; odd number of gripper changes |
| 78 | 7 grasp cycles where neighbours have 10; parts appear to pass on the belt |
| 129 | 1.78 s gap in the recording |
| 117 | 0.72 s gap |
| 103, 120, 123, 125 | Arm idle for 4–6 s in one stretch |
| 147–152 | Much longer (63–74 s, 16–19 grasp cycles); probably a different object count |
| 48–56 | Only 6–7 grasp cycles; the protocol seems to change at episode 57 |

Still to do:

1. Confirm recording has stopped, so the episode list is final.
2. Audit task success, which has never been done. Build one contact sheet per episode (first, middle and last frame of the middle camera) and count hole saws in the case against grasp cycles.
3. Decide on each flagged episode: keep, trim or drop. Proposed default: drop 50, 78 and 129; trim the idle stretches; keep 147–152 if the task text still describes them.
4. Done: gripper direction confirmed (recorder uses 1 = open).
5. Check the forward-kinematics pose on a few frames against the scene, including the base frame for the conveyor mounting.

## 6. Conversion

1. Resample every episode to an exact 15 Hz grid using the real `timestamp` values: interpolate joints and gripper, and pick the nearest video frame for each grid time. Do not drop every other frame, which would give about 14.3 Hz labelled as 15.
2. Compute the left end-effector pose for state (follower joints) and action (leader joints) by forward kinematics, in metres in one fixed rig frame. Store it as `[x, y, z, roll, pitch, yaw, gripper]`, and fill the right-arm slots with the constant parked pose (decision 2).
3. Flip the gripper to 0 = open, 1 = closed (`1 - value`) in the pose columns, for both arms. Keep the raw joint columns in the recorder convention.
4. Keep frames at the native 640x480 and encode H.264, yuv420p. Check that the hole saws and case survive a 16:10 crop in the overhead camera.
5. Write features: `observation.state` and `action` (14-D, left then right), `observation.joint_state` and `action.joint` (7-D left-arm joints), and `observation.images.top_left`, `wrist_left`, `wrist_right`.
6. Task text: `Pick up the hole saws from the moving conveyor belt and place them in the red case` (no closing period).
7. Write stats, plus a README with the source episode ID for every index, the held-out list, the camera roles and the resampling method.

This needs a new converter. `scripts/convert_to_lerobot.py` writes v2.1 at a fixed 30 fps with joints only.

## 7. Verification before upload

- Row `i` timestamp equals `i/15` for every episode.
- Resampled joints match the originals at the same wall-clock time (interpolation error near zero).
- Video frame count equals row count per episode and camera.
- A frame from each camera at a known grasp time shows the gripper closing.
- The dataset loads with the lerobot version Praxis uses, and one training window (93 frames) loads end to end.

## 8. Upload

- Push to `orchestrallabs/holesaw-conveyor-three-camera`, private, replacing the current tree.
- Re-download and repeat the checks in section 7 against the Hub copy.
- Disk: 72 GB free now. The converted set should be a few GB; the PNG staging in the old converter is what used the space last time.

## 9. Open risks

- There is no published way to fine-tune from the YAM checkpoint, and none for a single arm on it. The dataset can be made to match the checkpoint's inputs, but the training code is the open question.
- The Praxis requirements above are second-hand until someone runs the code. The cheapest proof is to run the YAM checkpoint's own input checks (`praxis/bimanual.py`) on one converted episode.
- Forward kinematics depends on the right YAM model and rig frame. Runway's YAM data uses its own rig frame; ours will differ, which a fine-tune has to absorb.
- A constant parked right arm is outside what the checkpoint saw in training.
- Compute for a YAM fine-tune is not stated. Inference alone needs one 80 GB GPU.
