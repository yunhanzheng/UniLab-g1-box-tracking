# Motion Preprocessing

This directory contains scripts for preprocessing motion data for motion tracking tasks.

## BONES-SEED CSV Replay

The `replay_bones_seed_csv.py` script replays local BONES-SEED G1 CSV clips
directly in the MuJoCo viewer.

### Input Format

The replay script expects a fixed 36-column layout:
- `Frame`
- `root_translateX/Y/Z`
- `root_rotateX/Y/Z`
- 29 `*_joint_dof` columns that map directly to G1 MuJoCo joint names

The script assumes:
- `root_translate*` is in centimeters and converts it to meters
- `root_rotate*` is in degrees
- `*_joint_dof` is in degrees

### Usage

```bash
# Replay the whole flip dataset
uv run scripts/motion/replay_bones_seed_csv.py

# Replay one clip
uv run scripts/motion/replay_bones_seed_csv.py \
  --input src/unilab/assets/motions/g1/flip/flip_090_001__A304.csv

# Validate parsing without opening the viewer
uv run scripts/motion/replay_bones_seed_csv.py --dry-run
```

### Controls

- `Space`: pause / resume
- `[`: previous CSV in playlist
- `]`: next CSV in playlist

## BONES-SEED CSV to NPZ

The `bones_seed_csv_to_npz.py` script converts local G1 flip CSV clips into NPZ
files with precomputed forward kinematics.

### Output Format

Generated NPZ files contain:
- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

### Usage

```bash
# Convert the whole flip dataset into src/unilab/assets/motions/g1/flip_npz
uv run scripts/motion/bones_seed_csv_to_npz.py

# Convert one clip next to a chosen output file
uv run scripts/motion/bones_seed_csv_to_npz.py \
  --input src/unilab/assets/motions/g1/flip/flip_090_001__A304.csv \
  --output temp/flip_090_001__A304.npz

# Validate inputs without exporting
uv run scripts/motion/bones_seed_csv_to_npz.py --dry-run
```

## CSV to NPZ Conversion

The `csv_to_npz.py` script converts motion data from CSV format to NPZ format with precomputed forward kinematics.

### Input Format

CSV files should contain motion data in Unitree's generalized coordinate convention:
- Columns 0-2: Base position (x, y, z)
- Columns 3-6: Base quaternion (x, y, z, w) - will be converted to wxyz internally
- Columns 7+: Joint angles (29 joints for G1)

### Output Format

NPZ files contain:
- `fps`: Frame rate (integer)
- `joint_pos`: Joint positions (N_frames × N_joints)
- `joint_vel`: Joint velocities (N_frames × N_joints)
- `body_pos_w`: Body positions in world frame (N_frames × N_bodies × 3)
- `body_quat_w`: Body quaternions in world frame (N_frames × N_bodies × 4, wxyz)
- `body_lin_vel_w`: Body linear velocities (N_frames × N_bodies × 3)
- `body_ang_vel_w`: Body angular velocities (N_frames × N_bodies × 3)

### Usage

```bash
# Basic usage
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50

# With custom model file
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50 \
  --model_file path/to/model.xml

# Process specific line range
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50 \
  --line_range 100 500
```

### Parameters

- `--input_file`: Path to input CSV file (required)
- `--output_file`: Path to output NPZ file (required)
- `--input_fps`: Frame rate of input CSV (default: 30)
- `--output_fps`: Desired output frame rate (default: 50)
- `--model_file`: MuJoCo model file (default: G1 flat scene)
- `--line_range`: Line range to process [start, end] (optional)

### Notes

- The script uses LERP for position interpolation and SLERP for quaternion interpolation
- Velocities are computed using numerical differentiation
- Forward kinematics is computed using MuJoCo for all bodies
- The output FPS should match the control frequency of your training environment (typically 50 Hz)

## Holosoma Lifting → G1 Box Tracking

Pipeline for holosoma-retargeted lifting motion with a synthetic box + platform
trajectory, used by `g1_box_tracking` (FlashSAC + Motrix).

### Files

| File | Description |
|------|-------------|
| `scripts/motion/lifting.npz` | Holosoma output (`robot_retarget.py` + `convert_data_format_mj.py`) |
| `scripts/motion/lifting_unilab.npz` | Robot motion remapped to UniLab G1 body-id layout |
| `scripts/motion/lifting_box_platform.npy` | Synthetic box/platform trajectory sidecar |
| `scripts/motion/lifting_unilab_box.npz` | Final NPZ with `object_*` keys for training/replay |
| `src/unilab/assets/robots/g1/scene_flat_with_largebox.xml` | Scene: floor, box, platform |

Default keyframes in `build_lifting_box_platform.py`: pickup frame **55**, place frame **126**.

### 1. Remap holosoma NPZ → UniLab

```bash
uv run python scripts/motion/remap_lifting_to_unilab.py \
  -i scripts/motion/lifting.npz \
  -o scripts/motion/lifting_unilab.npz \
  --model-xml src/unilab/assets/robots/g1/g1_sphere_hand.xml
```

### 2. Build box/platform trajectory

```bash
uv run python scripts/motion/build_lifting_box_platform.py \
  --lifting-npz scripts/motion/lifting_unilab.npz \
  --output scripts/motion/lifting_box_platform.npy \
  --pickup-frame 55 \
  --place-frame 126
```

### 3. Merge into final training NPZ

```bash
uv run python scripts/motion/build_lifting_boxtracking_npz.py \
  --lifting-npz scripts/motion/lifting_unilab.npz \
  --trajectory-npy scripts/motion/lifting_box_platform.npy \
  --output scripts/motion/lifting_unilab_box.npz
```

Regenerate all three outputs in one go:

```bash
uv run python scripts/motion/remap_lifting_to_unilab.py \
  -i scripts/motion/lifting.npz \
  -o scripts/motion/lifting_unilab.npz \
  --model-xml src/unilab/assets/robots/g1/g1_sphere_hand.xml

uv run python scripts/motion/build_lifting_box_platform.py \
  --lifting-npz scripts/motion/lifting_unilab.npz \
  --output scripts/motion/lifting_box_platform.npy \
  --pickup-frame 55 \
  --place-frame 126

uv run python scripts/motion/build_lifting_boxtracking_npz.py \
  --lifting-npz scripts/motion/lifting_unilab.npz \
  --trajectory-npy scripts/motion/lifting_box_platform.npy \
  --output scripts/motion/lifting_unilab_box.npz
```

After changing platform/box placement, update `scene_flat_with_largebox.xml` to match.

### 4. Replay (MuJoCo)

MuJoCo-only open-loop replay of the reference motion (no checkpoint required):

```bash
uv run scripts/motion/replay_npz.py \
  --npz_file scripts/motion/lifting_unilab_box.npz \
  --model_file src/unilab/assets/robots/g1/scene_flat_with_largebox.xml
```

No loop:

```bash
uv run scripts/motion/replay_npz.py \
  --npz_file scripts/motion/lifting_unilab_box.npz \
  --model_file src/unilab/assets/robots/g1/scene_flat_with_largebox.xml \
  --no-loop
```

Slow motion:

```bash
uv run scripts/motion/replay_npz.py \
  --npz_file scripts/motion/lifting_unilab_box.npz \
  --model_file src/unilab/assets/robots/g1/scene_flat_with_largebox.xml \
  --speed 0.5
```

### 5. Train (FlashSAC)

Owner configs:
- MuJoCo: `conf/offpolicy/task/flashsac/g1_box_tracking/mujoco.yaml`
- Motrix: `conf/offpolicy/task/flashsac/g1_box_tracking/motrix.yaml`

Defaults (task owner configs inherit `conf/offpolicy/algo/flashsac.yaml` unless overridden):

| Setting | Default |
|---------|--------:|
| `algo.num_envs` | 2048 |
| `algo.max_iterations` | 25000 (g1 box tracking task) |
| `algo.save_interval` | 10000 |

Checkpoints: `logs/flash_sac/G1BoxTracking/<run_timestamp>_<backend>/model_<iteration>.pt`

Each checkpoint stores **learner weights + training step** (Holosoma-style, no replay
buffer). Files are small (tens of MB). Resume restores policy/optimizer state and
continues from the saved iteration; the replay buffer refills from scratch.

Motrix training:

```bash
uv run train --algo flashsac --task g1_box_tracking --sim motrix \
  +env.motion_file=/home/ubuntu22/data/UniLab/scripts/motion/lifting_unilab_box.npz \
  algo.max_iterations=1000000 \
  training.no_play=true
```

MuJoCo backend:

```bash
uv run train --algo flashsac --task g1_box_tracking --sim mujoco \
  +env.motion_file=/home/ubuntu22/data/UniLab/scripts/motion/lifting_unilab_box.npz \
  algo.max_iterations=25000
```

**Resume training.** Restores learner/optimizer state and continues from the saved
iteration. The replay buffer is rebuilt from scratch (same as Holosoma FastSAC).
Older weight-only and legacy checkpoints still load; checkpoints saved with the
previous full-replay format still restore replay when present.

Set `training.resume=true` with `algo.load_run=-1` for the latest run, or pass a
run folder name. Increase `algo.max_iterations` beyond the checkpoint iteration:

```bash
uv run train --algo flashsac --task g1_box_tracking --sim motrix \
  +env.motion_file=/home/ubuntu22/data/UniLab/scripts/motion/lifting_unilab_box.npz \
  training.resume=true \
  algo.load_run=-1 \
  algo.max_iterations=1000000 \
  training.no_play=true
```

Note: the train script still restarts the collector subprocess after Ctrl+C or
crash, but model weights and training step continue from the latest checkpoint.

Resume a specific run:

```bash
uv run train --algo flashsac --task g1_box_tracking --sim motrix \
  +env.motion_file=/home/ubuntu22/data/UniLab/scripts/motion/lifting_unilab_box.npz \
  algo.load_run=2026-06-10_16-07-48_motrix \
  algo.checkpoint=20000 \
  algo.max_iterations=1000000 \
  training.no_play=true
```

Eval / playback:

```bash
uv run eval --algo flashsac --task g1_box_tracking --sim motrix --load-run -1 \
  +env.motion_file=/home/ubuntu22/data/UniLab/scripts/motion/lifting_unilab_box.npz
```

**TensorBoard** events are written directly under the run directory (not `tb/`):

```bash
tensorboard --logdir logs/flash_sac/G1BoxTracking/<run_timestamp>_motrix
```

**FlashSAC + CUDA / Triton compile**

FlashSAC enables `torch.compile` by default. If Triton/gcc fails with
`InductorError`, set CUDA stub paths before training:

```bash
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/stubs:$LIBRARY_PATH
export CUDA_HOME=/usr/lib/cuda

uv run train --algo flashsac --task g1_box_tracking --sim motrix \
  +env.motion_file=/home/ubuntu22/data/UniLab/scripts/motion/lifting_unilab_box.npz \
  training.no_play=true
```

Or disable compile: `algo.use_compile=false`.

**Motrix quaternion reset error** — if training crashes with
`invalid quaternion [0,0,0,0]`, regenerate the NPZ files (steps 1–3 above).
