"""Open-loop NPZ motion replay in the Motrix interactive renderer.

Uses the same G1 motion-tracking / box-tracking scenes as training so you can
inspect reference motions with Motrix physics and visuals (sphere-hand box scene,
largebox object state, etc.).

Usage:
    uv run scripts/motion/replay_npz_motrix.py \\
      --npz_file scripts/motion/lifting_unilab_box.npz

    # No box / flat scene
    uv run scripts/motion/replay_npz_motrix.py \\
      --npz_file scripts/motion/lifting_unilab.npz

    # Slow motion, play once
    uv run scripts/motion/replay_npz_motrix.py \\
      --npz_file scripts/motion/lifting_unilab_box.npz --speed 0.5 --no-loop
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.motion_tracking.g1.box_tracking import (
    G1BoxTrackingEnv,
    G1BoxTrackingEnvCfg,
    _build_box_motion_reference_state,
)
from unilab.envs.motion_tracking.g1.tracking import (
    G1MotionTrackingEnv,
    G1MotionTrackingEnvCfg,
    _build_motion_reference_state,
)

_OBJECT_KEYS = (
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
)


def _npz_has_object(npz_file: str) -> bool:
    with np.load(npz_file) as data:
        return all(key in data for key in _OBJECT_KEYS)


def _disable_reset_randomization(cfg: Any) -> None:
    cfg.joint_position_range = (0.0, 0.0)
    for attr in ("x", "y", "z", "roll", "pitch", "yaw"):
        setattr(cfg.pose_randomization, attr, (0.0, 0.0))
        setattr(cfg.velocity_randomization, attr, (0.0, 0.0))


def _make_env(npz_file: str, model_file: str | None):
    registry.ensure_registries()
    has_object = _npz_has_object(npz_file)
    if has_object:
        cfg = G1BoxTrackingEnvCfg()
        default_scene = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_with_largebox.xml"
    else:
        cfg = G1MotionTrackingEnvCfg()
        default_scene = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"

    cfg.motion_file = npz_file
    _disable_reset_randomization(cfg)
    if model_file is not None:
        cfg.scene = SceneCfg(model_file=model_file)
    elif not Path(cfg.scene.model_file).is_file():
        cfg.scene = SceneCfg(model_file=str(default_scene))

    env_cls = G1BoxTrackingEnv if has_object else G1MotionTrackingEnv
    env = env_cls(cfg, num_envs=1, backend_type="motrix")
    return env, has_object


def replay(args: argparse.Namespace) -> None:
    npz_path = Path(args.npz_file)
    if not npz_path.is_file():
        raise SystemExit(f"NPZ file not found: {npz_path}")

    env, has_object = _make_env(str(npz_path), args.model_file)
    num_frames = int(env.motion_loader.num_frames)
    fps = int(env.motion_loader.fps)
    frame_dt = 1.0 / max(fps, 1)

    print(f"Motion: {num_frames} frames @ {fps} Hz ({num_frames / fps:.1f}s)")
    print(f"Scene: {env.cfg.scene.model_file}")
    print(f"Backend: motrix ({'box' if has_object else 'flat'} tracking env)")
    print(f"Playback speed: {args.speed}x")

    if args.dry_run:
        print("Dry run OK — Motrix viewer was not opened.")
        env.close()
        return

    playback = {
        "frame": 0,
        "next_advance": time.perf_counter(),
        "finished": False,
    }
    env_ids = np.array([0], dtype=np.int32)

    def _apply_frame(frame_idx: int) -> None:
        motion_data = env.motion_loader.get_motion_at_frame(np.array([frame_idx], dtype=np.int32))
        if has_object:
            qpos, qvel = _build_box_motion_reference_state(env, env_ids, motion_data)
        else:
            qpos, qvel = _build_motion_reference_state(env, env_ids, motion_data)
        env._backend.set_state(env_ids, qpos, qvel)

    def initialize() -> int:
        _apply_frame(0)
        return 0

    def step(_: int) -> int:
        now = time.perf_counter()
        target_dt = frame_dt / max(args.speed, 1e-6)
        while now >= playback["next_advance"]:
            next_frame = playback["frame"] + 1
            if next_frame >= num_frames:
                if args.no_loop:
                    playback["finished"] = True
                    next_frame = num_frames - 1
                else:
                    next_frame = 0
            playback["frame"] = next_frame
            playback["next_advance"] += target_dt
        _apply_frame(playback["frame"])
        return playback["frame"]

    print("Opening Motrix viewer — close the window to exit.")
    if not args.no_loop:
        print("Looping enabled (use --no-loop to play once).")

    try:
        env.init_state()
        if args.no_loop:
            # Motrix renders ~60 Hz; scale steps so the full clip plays once.
            play_steps = int(np.ceil(num_frames * (60.0 / fps) / max(args.speed, 1e-6)))
        else:
            play_steps = None
        env.run_playback_mode(
            play_render_mode="auto",
            play_steps=play_steps,
            output_video=None,
            initialize=initialize,
            step=step,
        )
    finally:
        env.close()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay NPZ motion in Motrix viewer")
    parser.add_argument("--npz_file", type=str, required=True, help="Path to NPZ motion file")
    parser.add_argument("--model_file", type=str, default=None, help="Override scene XML path")
    parser.add_argument("--no-loop", action="store_true", help="Play once and exit")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate NPZ/scene loading without opening the Motrix viewer",
    )
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
