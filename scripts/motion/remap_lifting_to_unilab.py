#!/usr/bin/env python3
"""Remap a holosoma / full-body motion NPZ to UniLab G1 body/joint ordering.

Holosoma's ``convert_data_format_mj.py`` writes ``joint_pos`` as the full MuJoCo
``qpos`` vector: 7 root DOF + 29 actuated joints (shape T×36). UniLab expects
joint-only arrays (T×29) with root pose stored in ``body_pos_w`` / ``body_quat_w``.

Usage:
  python remap_lifting_to_unilab.py -i /path/to/lifting.npz -o out.npz \
    --model-xml src/unilab/assets/robots/g1/g1_sphere_hand.xml
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mujoco_motion_layout import expand_body_arrays_to_mujoco_body_ids

_ROOT_QPOS_DIM = 7
_ROOT_QVEL_DIM = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", dest="input", required=True)
    p.add_argument("-o", "--output", dest="output", required=True)
    p.add_argument("--model-xml", dest="model_xml", required=True)
    p.add_argument(
        "--motion-layout-xml",
        dest="motion_layout_xml",
        default=None,
        help="MuJoCo XML used to expand body arrays to body-id layout (default: --model-xml)",
    )
    return p.parse_args()


def extract_model_names(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    body_names = [b.get("name") for b in root.findall(".//body") if b.get("name")]
    joint_names = [j.get("name") for j in root.findall(".//joint") if j.get("name")]
    return body_names, joint_names


def map_and_reorder(src_names, dst_names):
    src_index = {n: i for i, n in enumerate(src_names)}
    return [src_index.get(n) for n in dst_names]


def reorder_array(src_arr, idx_map, dtype=np.float32):
    src = np.asarray(src_arr)
    if src.ndim < 2:
        raise ValueError("Expected array with at least 2 dims (T, N, ...)")
    t = src.shape[0]
    extra_shape = src.shape[2:]
    out = np.zeros((t, len(idx_map)) + extra_shape, dtype=dtype)
    for dst_i, src_i in enumerate(idx_map):
        if src_i is None:
            continue
        out[:, dst_i, ...] = src[:, src_i, ...].astype(dtype)
    return out


def reorder_joints(src_arr, idx_map):
    src = np.asarray(src_arr)
    t = src.shape[0]
    if src.ndim == 2:
        out = np.zeros((t, len(idx_map)), dtype=np.float32)
        for dst_i, src_i in enumerate(idx_map):
            if src_i is None:
                continue
            out[:, dst_i] = src[:, src_i].astype(np.float32)
        return out
    extra = src.shape[2:]
    out = np.zeros((t, len(idx_map)) + extra, dtype=np.float32)
    for dst_i, src_i in enumerate(idx_map):
        if src_i is None:
            continue
        out[:, dst_i, ...] = src[:, src_i, ...].astype(np.float32)
    return out


def strip_root_joint_arrays(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    num_joint_names: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove holosoma root free-joint prefix from joint arrays."""
    pos = np.asarray(joint_pos)
    vel = np.asarray(joint_vel)

    if pos.shape[1] == num_joint_names + _ROOT_QPOS_DIM:
        pos = pos[:, _ROOT_QPOS_DIM:]
    elif pos.shape[1] != num_joint_names:
        raise ValueError(
            f"Unexpected joint_pos width {pos.shape[1]} for {num_joint_names} joint names. "
            f"Expected {num_joint_names} (UniLab) or {num_joint_names + _ROOT_QPOS_DIM} (holosoma qpos)."
        )

    if vel.shape[1] == num_joint_names + _ROOT_QVEL_DIM:
        vel = vel[:, _ROOT_QVEL_DIM:]
    elif vel.shape[1] != num_joint_names:
        raise ValueError(
            f"Unexpected joint_vel width {vel.shape[1]} for {num_joint_names} joint names. "
            f"Expected {num_joint_names} (UniLab) or {num_joint_names + _ROOT_QVEL_DIM} (holosoma qvel)."
        )

    return pos, vel


def main():
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    model_xml = Path(args.model_xml)
    if not inp.exists():
        print(f"Input not found: {inp}")
        sys.exit(2)
    if not model_xml.exists():
        print(f"Model xml not found: {model_xml}")
        sys.exit(2)

    data = np.load(str(inp), allow_pickle=True)
    required = [
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    for k in required:
        if k not in data:
            print(f"Missing required key in input NPZ: {k}")
            sys.exit(3)

    src_body_names = data.get("body_names")
    src_joint_names = data.get("joint_names")
    if src_body_names is None or src_joint_names is None:
        print("Input NPZ must include 'body_names' and 'joint_names' arrays for mapping.")
        sys.exit(4)
    src_body_names = [str(x) for x in src_body_names]
    src_joint_names = [str(x) for x in src_joint_names]

    model_bodies, model_joints = extract_model_names(model_xml)
    body_map = map_and_reorder(src_body_names, model_bodies)
    joint_map = map_and_reorder(src_joint_names, model_joints)

    src_joint_pos, src_joint_vel = strip_root_joint_arrays(
        data["joint_pos"],
        data["joint_vel"],
        len(src_joint_names),
    )

    print(f"Source bodies: {len(src_body_names)}, Model bodies: {len(model_bodies)}")
    print(f"Mapped {sum(1 for i in body_map if i is not None)} bodies; {sum(1 for i in body_map if i is None)} missing")
    print(
        f"joint_pos: {data['joint_pos'].shape} -> actuated {src_joint_pos.shape} -> target ({src_joint_pos.shape[0]}, {len(model_joints)})"
    )

    body_pos = reorder_array(data["body_pos_w"], body_map, dtype=np.float32)
    body_quat = reorder_array(data["body_quat_w"], body_map, dtype=np.float32)
    body_lin = reorder_array(data["body_lin_vel_w"], body_map, dtype=np.float32)
    body_ang = reorder_array(data["body_ang_vel_w"], body_map, dtype=np.float32)

    layout_xml = Path(args.motion_layout_xml or args.model_xml)
    expanded = expand_body_arrays_to_mujoco_body_ids(
        {
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "body_lin_vel_w": body_lin,
            "body_ang_vel_w": body_ang,
        },
        model_bodies,
        layout_xml,
    )
    body_pos = expanded["body_pos_w"]
    body_quat = expanded["body_quat_w"]
    body_lin = expanded["body_lin_vel_w"]
    body_ang = expanded["body_ang_vel_w"]

    joint_pos = reorder_joints(src_joint_pos, joint_map)
    joint_vel = reorder_joints(src_joint_vel, joint_map)

    fps_arr = np.asarray(data["fps"])
    fps = int(fps_arr.reshape(-1)[0])

    np.savez_compressed(
        str(outp),
        fps=np.array([fps], dtype=np.int32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        body_names=np.array(model_bodies),
        joint_names=np.array(model_joints),
    )

    print(f"Wrote remapped NPZ to {outp} (body layout: {body_pos.shape[1]} MuJoCo bodies)")


if __name__ == "__main__":
    main()
