# Motion scripts

Preprocessing and replay for G1 motion tracking / box tracking.

## NPZ replay (reference motion)

Open-loop playback, no checkpoint. NPZ must include `fps`, `joint_pos`, `joint_vel`,
`body_pos_w`, `body_quat_w`, and (for box) `object_*` keys.


| Script                 | Backend                   | Scene                                                             |
| ---------------------- | ------------------------- | ----------------------------------------------------------------- |
| `replay_npz.py`        | MuJoCo viewer             | pass `--model_file`                                               |
| `replay_npz_motrix.py` | Motrix (same as training) | auto: box → `scene_flat_with_largebox.xml`, else `scene_flat.xml` |


**Box + Motrix (recommended before training):**

```bash
uv run scripts/motion/replay_npz_motrix.py \
  --npz_file scripts/motion/lifting_unilab_box.npz
```

**Box + MuJoCo:**

```bash
uv run scripts/motion/replay_npz.py \
  --npz_file scripts/motion/lifting_unilab_box.npz \
  --model_file src/unilab/assets/robots/g1/scene_flat_with_largebox.xml
```

**Robot-only (no box):**

```bash
uv run scripts/motion/replay_npz_motrix.py \
  --npz_file scripts/motion/lifting_unilab.npz
```

Common flags: `--speed 0.5`, `--no-loop`, `--dry-run` (Motrix only, no window).

Motrix replay needs a display (`DISPLAY`); macOS may require `mxpython` (same as eval).

---

## Holosoma lifting → box NPZ

Pipeline for `g1_box_tracking` (FlashSAC + Motrix).


| File                       | Role                                       |
| -------------------------- | ------------------------------------------ |
| `lifting.npz`              | Holosoma retarget output                   |
| `lifting_unilab.npz`       | Robot motion, UniLab body-id layout        |
| `lifting_box_platform.npy` | Synthetic box/platform trajectory          |
| `lifting_unilab_box.npz`   | Final NPZ with `object_*` for train/replay |


Default pickup / place frames: **55** / **126**. Scene:
`src/unilab/assets/robots/g1/scene_flat_with_largebox.xml`.

```bash
uv run python scripts/motion/remap_lifting_to_unilab.py \
  -i scripts/motion/lifting.npz \
  -o scripts/motion/lifting_unilab.npz \
  --model-xml src/unilab/assets/robots/g1/g1_sphere_hand.xml

uv run python scripts/motion/build_lifting_box_platform.py \
  --lifting-npz scripts/motion/lifting_unilab.npz \
  --output scripts/motion/lifting_box_platform.npy \
  --pickup-frame 55 --place-frame 126

uv run python scripts/motion/build_lifting_boxtracking_npz.py \
  --lifting-npz scripts/motion/lifting_unilab.npz \
  --trajectory-npy scripts/motion/lifting_box_platform.npy \
  --output scripts/motion/lifting_unilab_box.npz
```

After editing box/platform placement, update `scene_flat_with_largebox.xml` to match.

---

## FlashSAC: `g1_box_tracking`

Configs: `conf/offpolicy/task/flashsac/g1_box_tracking/{motrix,mujoco}.yaml`  
Checkpoints: `logs/flash_sac/G1BoxTracking/<timestamp>_<backend>/model_<iter>.pt`

**Train** (`training.no_play=true` — no window during training; playback runs after train if `no_play=false`):

```bash
uv run train --algo flashsac --task g1_box_tracking --sim motrix \
  +env.motion_file=scripts/motion/lifting_unilab_box.npz \
  algo.num_envs=2048 \
  algo.max_iterations=1000000 \
  algo.save_interval=10000 \
  training.no_play=true
```

**Resume** (latest run):

```bash
uv run train --algo flashsac --task g1_box_tracking --sim motrix \
  +env.motion_file=scripts/motion/lifting_unilab_box.npz \
  training.resume=true algo.load_run=-1 \
  algo.max_iterations=1000000 training.no_play=true
```

**Resume** (specific run + checkpoint iteration):

```bash
uv run train --algo flashsac --task g1_box_tracking --sim motrix \
  +env.motion_file=scripts/motion/lifting_unilab_box.npz \
  training.resume=true \
  algo.load_run=2026-06-10_16-07-48_motrix \
  +algo.checkpoint=20000 \
  algo.max_iterations=100000 training.no_play=true
```

**Eval** (opens Motrix window after load):

```bash
uv run eval --algo flashsac --task g1_box_tracking --sim motrix \
  algo.load_run=logs/flash_sac/G1BoxTracking/2026-06-17_15-06-09_motrix/model_200000.pt \
  +env.motion_file=scripts/motion/lifting_unilab_box.npz
```

Or `--load-run -1` for the newest run under `logs/flash_sac/G1BoxTracking/`.

**TensorBoard:** `uv run --with tensorboard tensorboard --logdir logs/flash_sac/G1BoxTracking --port 6006 --bind_all`

**Triton compile errors:** `export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/stubs:$LIBRARY_PATH CUDA_HOME=/usr/lib/cuda` or `algo.use_compile=false`.

---

## Other scripts

**BONES-SEED flip CSV → NPZ:** `bones_seed_csv_to_npz.py`  
**BONES-SEED CSV replay (MuJoCo):** `replay_bones_seed_csv.py` — `Space` pause, `[` / `]` prev/next clip  
**Unitree CSV → NPZ:** `csv_to_npz.py` — `--input_file`, `--output_file`, `--input_fps`, `--output_fps`

NPZ layout: `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`.