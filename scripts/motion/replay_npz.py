"""MuJoCo-only NPZ motion replay in the MuJoCo viewer.

Loads a preprocessed NPZ motion file and plays it back in the MuJoCo passive
viewer, setting qpos/qvel each frame so you can visually inspect the motion.
This replay path depends on the MuJoCo viewer/runtime and is not available for
Motrix-only workflows.

Usage:
    uv run scripts/motion/replay_npz.py --npz_file path/to/motion.npz

    # Custom model file
    uv run scripts/motion/replay_npz.py --npz_file motion.npz --model_file path/to/scene.xml

    # Loop playback (default)
    uv run scripts/motion/replay_npz.py --npz_file motion.npz

    # Play once and exit
    uv run scripts/motion/replay_npz.py --npz_file motion.npz --no-loop

    # Slow-motion (0.5x speed)
    uv run scripts/motion/replay_npz.py --npz_file motion.npz --speed 0.5
"""

# pyright: reportAttributeAccessIssue=false, reportReturnType=false

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH

_ROOT_QPOS_DIM = 7
_ROOT_QVEL_DIM = 6

G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def load_npz(npz_file: str) -> dict[str, np.ndarray]:
    """Load NPZ motion file and return arrays as a dict."""
    data = np.load(npz_file)
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    payload = {
        "fps": fps,
        "joint_pos": data["joint_pos"],
        "joint_vel": data["joint_vel"],
        "body_pos_w": data["body_pos_w"],
        "body_quat_w": data["body_quat_w"],
        "body_lin_vel_w": data["body_lin_vel_w"],
        "body_ang_vel_w": data["body_ang_vel_w"],
    }
    if "body_names" in data:
        payload["body_names"] = [str(x) for x in data["body_names"]]
    for key in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):
        if key in data:
            payload[key] = data[key]
    return payload


def default_model_path() -> str:
    """Return path to the default G1 flat scene XML."""
    return str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_with_largebox.xml")


def _resolve_root_body_index(body_names: list[str] | None, body_pos_w: np.ndarray) -> int:
    if body_names:
        if "pelvis" in body_names:
            return body_names.index("pelvis")
        if "world" in body_names:
            return body_names.index("pelvis") if "pelvis" in body_names else 1
    # UniLab exports omit world and store pelvis at index 0.
    if body_pos_w.shape[1] <= 31:
        return 0
    return 1


def _resolve_joint_columns(joint_pos: np.ndarray, joint_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return actuated joint arrays and whether root should come from joint_pos."""
    num_joints = len(G1_JOINT_NAMES)
    if joint_pos.shape[1] == num_joints + _ROOT_QPOS_DIM:
        return joint_pos[:, _ROOT_QPOS_DIM:], joint_vel[:, _ROOT_QVEL_DIM:], True
    return joint_pos, joint_vel, False


def replay(args):
    motion = load_npz(args.npz_file)
    fps = motion["fps"]
    joint_pos, joint_vel, root_in_joint_pos = _resolve_joint_columns(motion["joint_pos"], motion["joint_vel"])
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]
    body_names = motion.get("body_names")
    root_body_id = _resolve_root_body_index(body_names, body_pos_w)
    num_frames = joint_pos.shape[0]
    dt = 1.0 / fps

    object_pos_w = motion.get("object_pos_w")
    object_quat_w = motion.get("object_quat_w")

    print(f"Motion: {num_frames} frames @ {fps} Hz ({num_frames / fps:.2f}s)")
    print(f"Joints: {joint_pos.shape[1]}, Bodies: {body_pos_w.shape[1]}, root_body_id={root_body_id}")
    print(f"Playback speed: {args.speed}x")

    model_file = args.model_file or default_model_path()
    print(f"Model: {model_file}")

    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)

    joint_qpos_adr = []
    joint_qvel_adr = []
    for name in G1_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            print(f"Warning: joint '{name}' not found in model, skipping")
            joint_qpos_adr.append(None)
            joint_qvel_adr.append(None)
        else:
            joint_qpos_adr.append(model.jnt_qposadr[jnt_id])
            joint_qvel_adr.append(model.jnt_dofadr[jnt_id])

    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "largebox_joint")
    object_qpos_adr = model.jnt_qposadr[object_joint_id] if object_joint_id >= 0 else None

    def set_frame(frame_idx: int):
        if root_in_joint_pos:
            data.qpos[0:7] = motion["joint_pos"][frame_idx, :_ROOT_QPOS_DIM]
        elif body_pos_w.shape[1] > root_body_id:
            data.qpos[0:3] = body_pos_w[frame_idx, root_body_id]
            data.qpos[3:7] = body_quat_w[frame_idx, root_body_id]

        for j in range(min(joint_pos.shape[1], len(G1_JOINT_NAMES))):
            if joint_qpos_adr[j] is not None:
                data.qpos[joint_qpos_adr[j]] = joint_pos[frame_idx, j]
            if joint_qvel_adr[j] is not None:
                data.qvel[joint_qvel_adr[j]] = joint_vel[frame_idx, j]

        if object_qpos_adr is not None and object_pos_w is not None and object_quat_w is not None:
            data.qpos[object_qpos_adr : object_qpos_adr + 3] = object_pos_w[frame_idx]
            data.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = object_quat_w[frame_idx]

        mujoco.mj_forward(model, data)

    print("Opening viewer — close window or press Esc to quit.")
    if not args.no_loop:
        print("Looping enabled (use --no-loop to play once).")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            t0 = time.perf_counter()

            set_frame(frame)
            viewer.sync()

            frame += 1
            if frame >= num_frames:
                if not args.no_loop:
                    frame = 0
                else:
                    print("Playback finished.")
                    while viewer.is_running():
                        time.sleep(0.05)
                    break

            target_dt = dt / args.speed
            elapsed = time.perf_counter() - t0
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Replay NPZ motion in MuJoCo viewer")
    parser.add_argument("--npz_file", type=str, required=True, help="Path to NPZ motion file")
    parser.add_argument("--model_file", type=str, default=None, help="MuJoCo XML model file")
    parser.add_argument("--no-loop", action="store_true", help="Play once and exit (default: loop)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    if not Path(args.npz_file).exists():
        print(f"Error: NPZ file not found: {args.npz_file}")
        return

    replay(args)


if __name__ == "__main__":
    main()
