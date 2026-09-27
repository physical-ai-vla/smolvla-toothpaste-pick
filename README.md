# SmolVLA Toothpaste Pick — single-object grasp-and-lift on an SO-101 arm

Fine-tuning of the SmolVLA policy [`chamborgir/smolvla_pickplace_20k`](https://huggingface.co/chamborgir/smolvla_pickplace_20k) on a small set of teleoperated demonstrations (51 episodes, local dataset `local/toothpaste_grasp`) for 30,000 steps, then running it on a real SO-101 6-DOF follower arm with the task prompt `"Grasp the toothpaste box and lift it up"`.

The repository contains the training config snapshot, the policy config, the inference and dataset-cleanup scripts, and **one recorded real-robot rollout in which the arm grasps and lifts the box**. It does not contain the dataset, the checkpoints, or a success-rate evaluation (see [Status / limitations](#status--limitations)).

---

## Result — one successful rollout (2026-03-25)

![grasp-and-lift rollout](inference_results/ep01_20260325_090611.gif)

| Item | Value |
|------|-------|
| Video | [`inference_results/ep01_20260325_090611.mp4`](inference_results/ep01_20260325_090611.mp4) (1280×480 side-by-side, 30 fps, ~19.6 s, includes a 3 s countdown) |
| Preview GIF | `inference_results/ep01_20260325_090611.gif` (480×180, 12 fps) |
| Policy | `outputs/train/toothpaste_from_20k/checkpoints/last` (not included in the repo) |
| Prompt | `"Grasp the toothpaste box and lift it up"` |
| Inference | Apple Silicon Mac (MPS), `scripts/run_policy.py`, 30 Hz control loop |

Left half: `camera1` (dataset key `up`); right half: `camera2` (dataset key `side`). This is a single hand-picked episode; other rollouts from the same session were not successful and are not included.

---

## Status / limitations

- **Evidence is a single successful episode.** Only one rollout video is included, and it was selected as the success case. Other episodes from the same session did not succeed.
- **No quantitative evaluation.** The repo contains no success-rate measurement, no trial log, and no evaluation over object positions. `eval_freq` appears in the config, but `env` is `null`, so no simulated evaluation was run during training.
- **Single task, single object, single prompt.** The dataset uses one task string; this repo does not test instruction following across different prompts or generalization to new objects/positions.
- **No data augmentation in this run.** `dataset.image_transforms.enable` is `false` in `configs/train_config.json`.
- **Dataset and checkpoints are not included.** The dataset statistics below come from the author's local LeRobot dataset metadata and cannot be verified from this repo alone.
- **Scripts are hardware/environment specific.** Camera indices, serial port, and `GRIPPER_SCALE` must be adapted to your setup; the scripts assume a LeRobot 0.4.x source checkout (`sys.path.insert(0, "src")`).

---

## 1. Pipeline overview

```
SO-101 leader–follower teleoperation
  → toothpaste_grasp dataset (51 ep, 29,925 frames, 2 cameras, 30 fps)
  → cleanup (delete bad episodes, fix camera swap in ep 8–14)

chamborgir/smolvla_pickplace_20k (HF; SmolVLA trained 20K steps on SO-100 pick-place)
  ─[base]─→ SmolVLA fine-tune
              30,000 steps / batch 16 / AdamW lr=1e-4, cosine decay
              train_expert_only=True, freeze_vision_encoder=True
  → outputs/train/toothpaste_from_20k/checkpoints/...

scripts/run_policy.py (Mac MPS)
  → SO-101 follower + 2 USB cameras, 30 Hz inference loop
  → recordings/ep{NN}_<timestamp>.mp4 (1280×480 side-by-side)
```

---

## 2. Dataset — `local/toothpaste_grasp` (not included)

LeRobot dataset format stored locally (~1.1 GB, not in git). Numbers below are from the author's local metadata.

| Item | Value |
|------|-------|
| episodes | 51 |
| frames | 29,925 (~16.6 min at 30 fps) |
| state / action dim | 6 (SO-101 joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) |
| cameras | `observation.images.up`, `observation.images.side` (480×640) — renamed to `camera1`/`camera2` at training time via `rename_map` |
| task prompt | `"Grasp the toothpaste box and lift it up"` |
| collection | SO-101 leader–follower teleoperation with LeRobot recording |

### Cleanup utilities (`scripts/`)

Both scripts read `DATASET_ROOT = ~/.cache/huggingface/lerobot/local/toothpaste_grasp`.

| Script | Purpose |
|--------|---------|
| `view_episodes.py` | Play episodes one by one and delete bad ones (removes parquet metadata rows and repackages the video chunks) |
| `fix_camera_swap.py` | Swap the `up` ↔ `side` streams for episodes 8–14, where the camera cables were swapped during recording (ffmpeg cut + re-encode + parquet metadata update) |

Both scripts modify the dataset in place; back it up first.

---

## 3. Training — `toothpaste_from_20k`

`configs/train_config.json` is the full LeRobot training config snapshot (local absolute paths replaced with placeholders). `configs/policy_config.json` is the SmolVLA policy config.

### Policy

| Item | Value |
|------|-------|
| base policy | `chamborgir/smolvla_pickplace_20k` |
| VLM backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| `train_expert_only` | `True` (only the action expert is updated; no LoRA/PEFT) |
| `freeze_vision_encoder` | `True` |
| `chunk_size` / `n_action_steps` | 50 / 50 |
| normalization | state/action MEAN_STD, visual IDENTITY |

### Optimizer / scheduler

| Item | Value |
|------|-------|
| optimizer | AdamW (betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10, grad_clip=10.0) |
| scheduler | cosine_decay_with_warmup, 1,000 warmup steps, 30,000 decay steps |
| lr | 1e-4 peak → 2.5e-6 |

### Run settings

| Item | Value |
|------|-------|
| steps | 30,000 |
| batch size | 16 |
| seed | 1000 |
| num_workers | 4 |
| save_freq | 5,000 |
| log_freq | 200 |

### Training command (reconstructed from the config)

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=chamborgir/smolvla_pickplace_20k \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --dataset.repo_id=local/toothpaste_grasp \
  --dataset.root=/path/to/toothpaste_grasp \
  --batch_size=16 \
  --steps=30000 \
  --seed=1000 \
  --num_workers=4 \
  --optimizer.type=adamw --optimizer.lr=1e-4 --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10.0 \
  --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=1000 --scheduler.num_decay_steps=30000 \
    --scheduler.peak_lr=1e-4 --scheduler.decay_lr=2.5e-6 \
  --output_dir=outputs/train/toothpaste_from_20k \
  --save_freq=5000 --log_freq=200
```

The config also contains a `rename_map` (`observation.images.up → observation.images.camera1`, `observation.images.side → observation.images.camera2`) so that the dataset camera keys match the base policy's inputs.

Checkpoints (`model.safetensors`, ~865 MB each) are not included.

---

## 4. Inference — `scripts/run_policy.py`

Runs the trained policy on a real SO-101 follower with two USB cameras at 30 Hz on a Mac (MPS). Settings are constants at the top of the script:

```python
POLICY_PATH     = "outputs/train/toothpaste_from_20k/checkpoints/last/pretrained_model"
NORM_STATS_PATH = "outputs/train/toothpaste_from_20k/checkpoints/last/pretrained_model"

CAMERA1_INDEX = 1   # dataset key "up"   -> model input "camera1"
CAMERA2_INDEX = 0   # dataset key "side" -> model input "camera2"
FLIP_CAM1_HORIZONTAL = False   # (and FLIP_CAM1_VERTICAL, FLIP_CAM2_*)

GRIPPER_SCALE = 1.2  # multiplier on the predicted gripper action (value used in the demo)

ROBOT_PORT     = "/dev/tty.usbmodemXXXXXXXXXXX"   # your follower's serial port
ROBOT_ID       = "my_awesome_follower_arm"
TASK           = "Grasp the toothpaste box and lift it up"
N_EPISODES     = 3
EPISODE_TIME_S = 100
FPS            = 30
DEVICE         = "mps"
```

### Run

```bash
# inside a LeRobot 0.4.x source checkout (the script adds ./src to sys.path)
cd <lerobot repo>
python /path/to/scripts/run_policy.py
```

### What it does

1. Connects the SO-101 follower (6 Feetech STS3215 servos) and both cameras.
2. Shows a live side-by-side preview window with a 3 s countdown.
3. Each step feeds `observation.state` (6 joint positions) and both RGB images to SmolVLA.
4. `policy.select_action` returns actions from a 50-step chunk; the gripper action is multiplied by `GRIPPER_SCALE` and sent to the robot.
5. Records each episode to `recordings/ep{NN}_<YYYYMMDD_HHMMSS>.mp4`.

Camera indices can swap depending on OS/USB hub order. Check the preview window and make sure `camera1` shows the same view as the dataset's `up` camera.

Dependencies: `lerobot` 0.4.x, `torch` (with MPS or CUDA), `opencv-python`, `numpy`; `pyarrow` and `ffmpeg` for the cleanup scripts.

---

## 5. Repository layout

```
smolvla-toothpaste-pick/
├── README.md
├── .gitignore
├── scripts/
│   ├── run_policy.py            # SO-101 + SmolVLA real-time inference (MPS)
│   ├── view_episodes.py         # episode viewer + deletion
│   └── fix_camera_swap.py       # camera swap fix for ep 8–14
├── configs/
│   ├── train_config.json        # LeRobot training config snapshot
│   └── policy_config.json       # SmolVLA policy config
└── inference_results/
    ├── ep01_20260325_090611.mp4 # the single successful rollout shown above
    └── ep01_20260325_090611.gif
```

---

## 6. Hardware

- SO-101 follower arm (6× Feetech STS3215, USB serial) and leader arm for teleoperation
- 2 USB cameras (640×480, 30 fps)
- Hugging Face Hub access for `chamborgir/smolvla_pickplace_20k` and `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

---

## 7. Credits

- Base policy: [`chamborgir/smolvla_pickplace_20k`](https://huggingface.co/chamborgir/smolvla_pickplace_20k)
- VLM backbone: [`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- Framework: [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face)
- Robot: [SO-100 / SO-101](https://github.com/TheRobotStudio/SO-ARM100) (TheRobotStudio)

No license file is included in this repository.
