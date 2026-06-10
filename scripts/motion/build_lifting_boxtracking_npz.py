#!/usr/bin/env python3
"""Build a BoxMotionLoader-compatible NPZ from lifting.npz plus synthetic box/platform motion.

This keeps the robot data from the lifting reference motion and adds synthetic
object state so the motion can be previewed with the G1 box-tracking Motrix
pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mujoco_motion_layout import expand_body_arrays_to_mujoco_body_ids

G1_ASSETS = Path(__file__).resolve().parents[2] / "src" / "unilab" / "assets" / "robots" / "g1"
G1_BOX_SCENE_XML = G1_ASSETS / "scene_flat_with_largebox.xml"
G1_ROBOT_XML = G1_ASSETS / "g1_sphere_hand.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifting-npz", required=True, help="Robot-only lifting.npz")
    parser.add_argument("--trajectory-npy", required=True, help="Synthetic box/platform .npy")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument(
        "--motion-layout-xml",
        default=str(G1_BOX_SCENE_XML),
        help="MuJoCo XML used to expand body arrays to body-id layout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lifting_path = Path(args.lifting_npz)
    traj_path = Path(args.trajectory_npy)
    output_path = Path(args.output)

    with np.load(lifting_path, allow_pickle=True) as lift:
        traj = np.load(traj_path, allow_pickle=True).item()
        fps = int(np.asarray(lift["fps"]).reshape(-1)[0])
        box_pos_w = np.asarray(traj["box_pos_w"], dtype=np.float32)
        dt = 1.0 / max(fps, 1)
        payload = {
            "fps": np.array([fps], dtype=np.int32),
            "joint_pos": lift["joint_pos"].astype(np.float32),
            "joint_vel": lift["joint_vel"].astype(np.float32),
            "body_pos_w": lift["body_pos_w"].astype(np.float32),
            "body_quat_w": lift["body_quat_w"].astype(np.float32),
            "body_lin_vel_w": lift["body_lin_vel_w"].astype(np.float32),
            "body_ang_vel_w": lift["body_ang_vel_w"].astype(np.float32),
            "object_pos_w": box_pos_w,
            "object_quat_w": np.asarray(traj["box_quat_w"], dtype=np.float32),
            "object_lin_vel_w": np.gradient(box_pos_w, dt, axis=0).astype(np.float32),
            "object_ang_vel_w": np.zeros((box_pos_w.shape[0], 3), dtype=np.float32),
        }
        if "joint_names" in lift:
            payload["joint_names"] = lift["joint_names"]
        if "body_names" in lift:
            body_names = [str(x) for x in lift["body_names"]]
            payload["body_names"] = lift["body_names"]
            import mujoco

            layout_model = mujoco.MjModel.from_xml_path(str(args.motion_layout_xml))
            body_axis = payload["body_pos_w"].shape[1]
            if body_axis < layout_model.nbody:
                robot_model = mujoco.MjModel.from_xml_path(str(G1_ROBOT_XML))
                source_layout_xml = str(G1_ROBOT_XML) if body_axis == robot_model.nbody else None
                payload.update(
                    expand_body_arrays_to_mujoco_body_ids(
                        {
                            "body_pos_w": payload["body_pos_w"],
                            "body_quat_w": payload["body_quat_w"],
                            "body_lin_vel_w": payload["body_lin_vel_w"],
                            "body_ang_vel_w": payload["body_ang_vel_w"],
                        },
                        body_names,
                        args.motion_layout_xml,
                        source_model_xml=source_layout_xml,
                    )
                )

    # Align object velocities with the object trajectory length.
    payload["object_lin_vel_w"] = payload["object_lin_vel_w"].astype(np.float32)
    if payload["object_lin_vel_w"].shape[1] != 3:
        raise ValueError("object_lin_vel_w must have shape (T, 3)")

    np.savez_compressed(output_path, **payload)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
