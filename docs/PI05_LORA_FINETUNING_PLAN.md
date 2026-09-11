# Official $\pi_{0.5}$ Fine-Tuning Plan: YAM Red-Cap Task

This is the working plan for adapting the **official Physical Intelligence $\pi_{0.5}$ base model** to the 100 collected YAM/Gello demonstrations in `data/episodes/`.

Run two controlled experiments from the same official base and select by held-out metrics plus guarded robot rollouts:

- primary: apply LoRA to the 2B PaliGemma vision-language backbone while training the complete 300M action expert and action projections;
- comparison: fully fine-tune the 2B backbone, 300M action expert, and projections;
- use the official OpenPI JAX trainer and official base weights;
- train on the active left arm only;
- reconstruct the demonstrations using their real timestamps rather than the MP4 container frame rate.

The full run is a requested comparison, not an assertion that OpenPI requires full fine-tuning. It has higher overfitting risk on 100 demonstrations and must not receive an easier evaluation split.

---

## 1. Locked Decisions

| Item | Decision |
| :--- | :--- |
| Base weights | `gs://openpi-assets/checkpoints/pi05_base/params` |
| Training implementation | Official `Physical-Intelligence/openpi`, JAX path |
| Adaptations | (A) PaliGemma 2B LoRA + full 300M expert; (B) full-model fine-tuning |
| LoRA implementation | OpenPI's built-in `gemma_2b_lora` (rank 16, alpha 16) |
| Training hardware | One healthy 80 GB accelerator per concurrent run; use the best verified H100/A100 value available at launch |
| Robot output | Canonical single-arm layout: 6 joints + 1 gripper; this dataset binds it to left |
| Internal model width | 32 state/action dimensions, with OpenPI padding the 7 real dimensions |
| Dataset rate | Timestamp-derived, resampled to 15 Hz |
| Action horizon | 15 steps (1 second at 15 Hz) |
| Prompt | `pick up the red cap and place it in the black box` |
| Primary stopping window | Compare checkpoints at 2,500, 5,000, 7,500, and 10,000 steps |

The deterministic held-out source episode IDs are `6, 16, 26, 36, 46, 56, 66, 76, 86, 96`. Conversion produced 17,564 training frames and 1,993 evaluation frames. Both runs use random seed 42.

Do **not** use `helen9975/pi05-molmoact-yam` in this experiment. It belongs to a different training/checkpoint stack and would make this no longer a clean test of the official base model.

---

## 2. What Is Actually in the Dataset

The raw collection contains:

- 100 episodes and 25,874 control samples;
- approximately 21.7 minutes of timestamped demonstrations;
- `middle.mp4`, `left.mp4`, and `right.mp4` for every episode;
- `trajectory.npz` with `follower_joints`, `leader_actions`, timestamps, and tracking error;
- 14-dimensional state/action arrays: 7 dimensions per arm.

Important findings from the audit:

1. **The data is not actually 30 Hz.** The videos declare 30 FPS, but the number of video frames follows the control-loop samples. Episodes 1-60 are approximately 23.1 Hz; episodes 61-100 are approximately 13.9 Hz. The global timestamp-derived rate is approximately 19.8 Hz.
2. **The rate change is software-related, not evidence of a data failure.** Collection paused between episodes 60 and 61. During that pause, commit `3bd960c` added synchronous Rerun logging, including three per-step JPEG encodes, to the control loop. That is the likely cause of the slower second block. The mechanical work merely coincided with the boundary.
3. **The MP4 streams and trajectory lengths match.** This means samples can be aligned by frame index and assigned the corresponding trajectory timestamp; MP4 playback timing must not be treated as ground truth.
4. **Only the left arm performs the task.** Dimensions 7-13 are effectively a held right-arm pose, and right gripper dimension 13 is identically zero. Including these dimensions in fresh normalization statistics would create degenerate channels.
5. **YAM's native gripper convention is confirmed:** `0 = closed`, `1 = open`. The default resting state is closed. This is also recorded in `configs/yam_encoder_hw.yaml`.
6. The three cameras are external views, not true wrist cameras. They can still occupy the model's three image slots, but the mapping must be identical during conversion and deployment.

Never rewrite or delete the raw episodes. Write the converted dataset to a separate directory.

---

## 3. Data Conversion

Create a custom conversion script based on OpenPI's LeRobot examples. The output may live locally; uploading it to Hugging Face is optional.

### 3.1 Episode-level split

Split complete episodes, not individual frames:

- 90 training episodes;
- 10 held-out episodes;
- represent both collection-rate groups in each split;
- distribute visible object starting positions and execution styles across the split where possible;
- record the episode IDs in a versioned manifest so every run uses the same split.

The held-out split is useful for visual/open-loop inspection, but the deciding metric remains real robot success rate.

### 3.2 Reconstruct a uniform 15 Hz timeline

For each episode:

1. Load the timestamps from `trajectory.npz`.
2. Create target times at 1/15-second intervals within the episode's recorded interval.
3. Interpolate continuous joint states and joint-position commands onto the target times.
4. Use nearest-time source images for each target time.
5. Treat gripper values as absolute values; use nearest-time or zero-order-hold sampling rather than converting them to deltas.
6. Record the source-frame index and timestamp offset for auditing.

Fifteen hertz is close to the slower 40-episode block and avoids pretending that the later episodes contain temporal resolution they do not have. It should produce roughly 19,500 converted samples.

Also trim obvious setup idle time at the start and excessive idle time at the end. Preserve approximately 0.5-1.0 seconds after task completion so the policy learns to stop and hold.

### 3.3 Store only the active control dimensions

Use:

```python
state = follower_joints[..., :7]
action = leader_actions[..., :7]
```

The seven dimensions are:

```text
[left_joint_0, ..., left_joint_5, left_gripper]
```

The robot runtime, not the neural policy, must continue holding the inactive arm in its safe resting pose. A future right-only binding uses the same seven policy dimensions `0-6`; it must not move the active vector to bimanual dimensions `7-13`.

### 3.4 Camera mapping

Use this mapping consistently:

| OpenPI model slot | Dataset source | Physical meaning |
| :--- | :--- | :--- |
| `base_0_rgb` | `middle.mp4` | Primary external/table view |
| `left_wrist_0_rgb` | `left.mp4` | Auxiliary external left-side view |
| `right_wrist_0_rgb` | `right.mp4` | Auxiliary external right-side view |

OpenCV decodes images as BGR. Convert them to RGB before writing arrays or passing observations to OpenPI. All three image masks should be true when all views are present.

---

## 4. YAM-Specific OpenPI Transforms

Do not reuse `LeRobotAlohaDataConfig` unchanged. Its optional adaptation logic is specifically for Trossen/Aloha geometry and gripper encoding. Add `YamInputs`, `YamOutputs`, and `LeRobotYamDataConfig` with matching training and inference behavior.

Apply transforms in this semantic order:

### Model input and training actions

1. Repack LeRobot fields and camera keys.
2. Select the first seven state/action dimensions.
3. Convert YAM gripper values into the official $\pi$ convention:

   ```python
   pi_gripper = 1.0 - yam_gripper
   ```

   This conversion applies to state dimension 6 and action dimension 6.
4. Convert only joint action dimensions 0-5 from absolute targets to deltas from the current state.
5. Leave gripper dimension 6 absolute.
6. Compute/apply quantile normalization.
7. Let OpenPI pad state/actions from 7 to the model's internal 32 dimensions.

The delta mask is:

```python
transforms.make_bool_mask(6, -1)
```

### Policy output

OpenPI's policy pipeline applies these in the inverse semantic order:

1. unnormalize the padded model output (OpenPI pads the seven-channel statistics safely to 32 dimensions);
2. add current state back to joint dimensions 0-5;
3. leave gripper dimension 6 absolute;
4. discard padded action dimensions and retain dimensions 0-6;
5. convert the gripper back with `yam_gripper = 1.0 - pi_gripper`;
6. enforce safety limits before sending commands to the robot.

The inversion must occur **before normalization during training** and **after unnormalization during inference**. Add round-trip tests for closed (`0` native), open (`1` native), and an intermediate value.

---

## 5. Model and Training Configuration

The exact model object used for `get_freeze_filter()` must match the model object in `TrainConfig`.

```python
yam_model = pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,       # Required shape for the official pi0.5 checkpoint.
    action_horizon=15,   # One second at the converted 15 Hz rate.
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m",  # Full expert, not expert LoRA.
)

TrainConfig(
    name="pi05_yam_red_cap_lora",
    model=yam_model,
    data=LeRobotYamDataConfig(
        repo_id="npow/yam-gello-red-cap-100-15hz-train",
        default_prompt="pick up the red cap and place it in the black box",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=2.5e-5,
        decay_steps=10_000,
        decay_lr=2.5e-6,
    ),
    optimizer=_optimizer.AdamW(),
    freeze_filter=yam_model.get_freeze_filter(),
    ema_decay=None,
    batch_size=32,
    num_train_steps=10_000,
    save_interval=500,
    keep_period=2_500,
    log_interval=50,
)
```

Notes:

- `action_dim=32` describes the checkpoint/model tensor shape. It does **not** mean the YAM policy controls 32 physical dimensions.
- `discrete_state_input` defaults to true when `pi05=True`; spelling it out is optional.
- OpenPI's `gemma_2b_lora` currently uses rank 16 and alpha 16, not alpha 32.
- Use OpenPI's default AdamW weight decay (`1e-10`) for the baseline. Do not introduce `0.01` across all trainable weights without a controlled comparison.
- JAX OpenPI is required for this LoRA run. OpenPI's PyTorch trainer does not currently support LoRA.
- If batch 32 is unstable or out of memory, retry batch 16. Compare runs in examples processed as well as raw step count.

### Controlled full-fine-tuning comparison

The full configuration uses `pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=15)`, no freeze filter, and the same data, seed, batch size, LR schedule, EMA setting, save cadence, and 10,000-step budget. Both configurations load only `gs://openpi-assets/checkpoints/pi05_base/params` and share the exact same normalization-statistics file.

LoRA plus a full expert remains the lower-risk prior for this small dataset because it can learn YAM kinematics without freely moving every vision-language weight. Full fine-tuning is valid and fits an 80 GB H100, but it uses more than 70 GB and may overfit. Decide from held-out open-loop metrics and robot task success, not training loss.

---

## 6. Normalization Statistics

Compute fresh statistics from the converted **training split** after the YAM input transforms and delta-action transform have been applied.

Do not point `AssetsConfig` at a checkpoint directory for the first stats run. With the default local asset base directory, run:

```bash
cd /home/npow/code/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_yam_red_cap_lora
```

The current OpenPI script writes to:

```text
/home/npow/code/openpi/assets/pi05_yam_red_cap_lora/<repo-id>/norm_stats.json
```

Before training, inspect the resulting arrays and confirm:

- exactly seven unpadded state/action channels are represented in the dataset transforms;
- all values are finite;
- joint delta quantiles are plausible;
- the transformed gripper follows `0 = open`, `1 = closed`;
- no zero-variance right-arm dimensions remain;
- q01/q99 do not reveal a small number of corrupt command jumps dominating the scale.

---

## 7. Vast.ai Hardware Selection

Use two verified 80 GB H100 accelerators when suitable interruptible offers are available, one per run. Re-run the search immediately before renting because Vast.ai prices and availability are dynamic.

```bash
vastai search offers \
  'gpu_name in ["H100 SXM","H100 NVL","H100 PCIE","A100 SXM4","A100 PCIE"] num_gpus=1 gpu_ram>=75 reliability>0.98 verified=true rentable=true direct_port_count>=1 disk_space>=150 cuda_vers>=12.1' \
  -o 'dlperf_usd-'
```

Selection rules:

1. Sort by `dlperf_usd`, not hourly price alone.
2. Prefer H100 SXM when it leads the value ranking and has at least 80 GB VRAM.
3. Otherwise take the best A100 SXM4 80 GB offer; avoid the 40 GB A100 for possible full-tune follow-up runs.
4. Require reliability above 98%, adequate download/upload bandwidth, at least 150 GB fast local disk, and enough allowed rental duration.
5. Use interruptible pricing near $1/hour, save recovery checkpoints every 500 steps, and automatically resume after eviction. Fall back to on-demand only if repeated evictions prevent progress.

On 2026-09-10, the live verified market included an 80 GB H100 SXM offer at approximately **$0.87/hour interruptible** with 99.35% reliability; another was approximately $0.66/hour at 97.74% reliability. Comparable on-demand H100 inventory started around $1.97/hour, while a $2.94/hour listing was not cost-effective. These are snapshots, not durable quotes.

The comparison began on Vast offer `36742497` / instance `50541903`: two H100 SXM 80 GB GPUs, 500 GB disk, reliability 99.91%, and OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`. Its all-in instance rate is $2.7526/hour. One GPU proved defective: it reached 92 C, fell to a 345 MHz SM clock, and was removed from sustained work. Full fine-tuning remains healthy on the other H100 at approximately 1.5 seconds/step.

The first LoRA replacement, offer `49402836` / instance `50543765`, was a single interruptible H100 SXM 80 GB at $1.0333/hour all-in. It ran normally at approximately 1.3 seconds/step but was preempted near step 350, before the first step-500 recovery checkpoint. Its disk is retained while stopped at $0.0833/hour; repeated restart requests have remained queued despite a bid above the displayed minimum.

LoRA was therefore restarted from the official base on on-demand offer `49067611` / instance `50546595`: one A100 SXM4 80 GB, 150 GB disk, 99.49% reliability, and $1.3257/hour all-in. The replacement passed its initial health check and advances at approximately 3.0 seconds/step at 62-63 C. The intended cost/time optimization is to hand its latest complete checkpoint to the healthy H100 after the full run finishes, then release the A100. Record the final handoff step, measured rates, metrics, and actual spend in the run log.

Expected wall-clock range:

| Phase | H100 80 GB | A100 80 GB |
| :--- | :---: | :---: |
| Setup, download, JAX compilation | 30-90 min | 30-90 min |
| 10,000-step LoRA + full-expert run | approximately 2-5 hr | approximately 4-8 hr |
| 10,000-step full-model run | approximately 3-7 hr | approximately 6-12 hr |
| First guarded robot evaluation | 1-3 hr | 1-3 hr |

These are planning ranges, not guarantees. Benchmark the first 100-200 post-compilation steps and replace the estimate with:

```text
remaining_hours = remaining_steps * median_seconds_per_step / 3600
```

---

## 8. Commands

Pin a clean OpenPI checkout or worktree before editing it. The existing local checkout has unrelated changes, so do not blindly pull or overwrite it.

After adding the YAM conversion, transforms, and config:

```bash
cd /home/npow/code/openpi

uv run scripts/compute_norm_stats.py \
  --config-name pi05_yam_red_cap_lora

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py pi05_yam_red_cap_lora \
  --exp-name=yam_red_cap_lora_run01 \
  --overwrite

CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py pi05_yam_red_cap_full \
  --exp-name=yam_red_cap_full_run01 \
  --overwrite
```

If the 80 GB instance still runs out of memory, first try `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, then lower the batch size. Do not change multiple training variables at once.

Serve a selected checkpoint with the same config and transforms:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_yam_red_cap_lora \
  --policy.dir=checkpoints/pi05_yam_red_cap_lora/yam_red_cap_run01/7500
```

The Servo-facing contract is independent of both physical side and training method:

```bash
servo serve pi0.5 \
  --embodiment yam-single \
  --checkpoint /absolute/path/to/promoted/checkpoint
```

`yam-single` always exchanges a compact seven-dimensional vector. The robot binding chooses `left` or `right`; LoRA versus full remains checkpoint/runtime metadata. This checkpoint requires the `top`, `left`, and `right` camera roles. A request with only `top` and `left` must fail compatibility until a missing-view mask has been deliberately trained and evaluated.

---

## 9. Evaluation and Deployment

Training loss is a debugging signal, not the final model-selection metric.

For each candidate checkpoint from both runs:

1. Plot predicted action chunks against held-out demonstrations before connecting the robot.
2. Confirm gripper direction, joint ordering, delta reconstruction, and numerical bounds.
3. Run at reduced speed with an operator at the emergency stop.
4. Test multiple red-cap starting positions represented in the data.
5. Record successes, grasp failures, placement failures, collisions/near-collisions, and timeouts.

Use at least 10 repeatable rollout attempts per serious candidate before declaring one better. Prefer task success and safe stopping behavior over the lowest training loss.

The policy predicts 15 actions, but deployment should be receding-horizon:

- request a new chunk approximately every 5 policy steps;
- execute policy steps semantically at 15 Hz;
- if the low-level controller runs faster, interpolate safe joint targets between policy steps;
- hold position on network/model timeout;
- keep joint-position, joint-velocity, acceleration, workspace, and gripper limits outside the neural network.

---

## 10. Pre-Flight Checklist

### Data

- [ ] Raw episodes are immutable and separately backed up.
- [ ] Conversion uses `trajectory.npz` timestamps, not MP4-declared FPS.
- [ ] Converted episodes are uniformly 15 Hz and retain episode boundaries.
- [ ] Train/evaluation episode IDs are saved in a manifest.
- [ ] Only left-arm dimensions 0-6 are presented to normalization/training.
- [ ] Images are RGB and camera mapping matches live inference.
- [ ] Idle trimming was visually spot-checked.

### Semantics

- [ ] Native YAM convention is tested as `0 = closed`, `1 = open`.
- [ ] Model-facing gripper conversion is tested as `g_pi = 1 - g_yam`.
- [ ] Joint dimensions 0-5 are deltas; gripper dimension 6 is absolute.
- [ ] A transform round-trip reproduces representative native commands.
- [ ] Right arm is held by a separate safe controller.
- [ ] Left-only and right-only robot bindings both map the active arm to policy dimensions 0-6.
- [ ] Camera names are not used to infer which physical arm receives actions.

### Model and run

- [ ] The official `pi05_base/params` checkpoint is the only initializer.
- [ ] Model `action_dim` remains 32 and physical output is sliced to 7.
- [ ] Model config and freeze-filter config match exactly.
- [ ] EMA is disabled in both runs so freezing is the principal comparison variable.
- [ ] Fresh quantile statistics load successfully.
- [ ] The OpenPI commit and dirty/clean status are recorded.
- [ ] Vast.ai offer details and hourly price are recorded.
- [ ] Recovery checkpoints are written every 500 steps and candidates are retained every 2,500 steps.

### Robot safety

- [ ] Open-loop plots pass before actuation.
- [ ] Joint/rate/workspace clamps are enabled outside the model.
- [ ] Timeout behavior holds position.
- [ ] First rollouts use reduced speed and a reachable emergency stop.

---

## 11. Main Gotchas

1. Trusting the MP4's 30 FPS metadata silently changes the task speed.
2. Sending native YAM gripper values directly to the official base reverses open and closed semantics.
3. Setting model `action_dim=7` or `14` can make the official checkpoint fail to load; keep 32 and pad/slice.
4. Normalizing the inactive right arm creates useless or degenerate statistics.
5. Treating the side cameras as literal wrist cameras is harmless only if the mapping remains consistent.
6. Computing stats before gripper inversion and joint-delta conversion makes training and inference disagree.
7. Batch size 32 is a starting point, not a promise; measure memory and throughput on the selected host.
8. Ten thousand steps at batch 32 is roughly 18 passes over the 17,564-sample training dataset. Later checkpoints can overfit, so evaluate earlier ones.
9. A successful held-out video prediction does not establish closed-loop robot success.
10. Synchronous image encoding/logging inside the control loop can reproduce the collection-rate problem in future data.

---

## 12. References

- [Official OpenPI repository and memory requirements](https://github.com/Physical-Intelligence/openpi)
- [OpenPI normalization-statistics guidance](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md)
- [OpenPI remote-inference/action-chunk guidance](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)
- [Vast.ai offer-search documentation](https://docs.vast.ai/cli/reference/search-instances)
- [Vast.ai pricing model](https://docs.vast.ai/guides/instances/pricing)
