#!/usr/bin/env python3
"""Build a synthetic box/platform trajectory for the lifting motion.

This script takes the robot-only lifting motion and synthesizes a simple box
trajectory plus a static platform trajectory.

Output format is a NumPy .npy file containing a dict with these keys:
  - fps: int
  - num_frames: int
  - pickup_frame: int
  - place_frame: int
  - box_pos_w: (T, 3)
  - box_quat_w: (T, 4)
  - platform_pos_w: (T, 3)
  - platform_quat_w: (T, 4)
  - box_size: (3,)
  - platform_size: (3,)

The box is stationary before pickup, follows the hands with a fixed relative offset
from pickup through place, and freezes in place after place.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

G1_SPHERE_HAND_XML = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "unilab"
    / "assets"
    / "robots"
    / "g1"
    / "g1_sphere_hand.xml"
)

DEFAULT_PLATFORM_FORWARD_DIST = 0.55
SPAWN_FORWARD_OFFSET = 0.1
PLATFORM_HALF_XYZ = (0.225, 0.15, 0.04)
# largebox uses an axis-aligned box geom in scene_flat_with_largebox.xml.
# Object freejoint quat stays identity; body origin is the box center.
BOX_HALF_XYZ = (0.05, 0.05, 0.05)
BOX_HALF_Z = BOX_HALF_XYZ[2]
OBJECT_REST_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

LEFT_WRIST_CANDIDATES = (
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
)
RIGHT_WRIST_CANDIDATES = (
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
)


def _extract_model_names(xml_path: Path) -> tuple[list[str], list[str]]:
    root = ET.parse(xml_path).getroot()
    body_names = [b.get("name") for b in root.findall(".//body") if b.get("name")]
    joint_names = [j.get("name") for j in root.findall(".//joint") if j.get("name")]
    return body_names, joint_names


def _load_motion(path: Path, model_xml: Path) -> dict[str, np.ndarray | list[str] | int]:
    with np.load(path, allow_pickle=True) as data:
        body_names = data.get("body_names")
        if body_names is None:
            body_names, _ = _extract_model_names(model_xml)
        else:
            body_names = [str(x) for x in body_names]
        return {
            "fps": int(np.asarray(data["fps"]).reshape(-1)[0]),
            "body_names": body_names,
            "body_pos_w": data["body_pos_w"].astype(np.float32),
            "body_quat_w": data["body_quat_w"].astype(np.float32),
        }


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return (vec / norm).astype(np.float32)


def _quat_apply_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    vx, vy, vz = vec
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float32,
    )


def _forward_xy(quat: np.ndarray) -> np.ndarray:
    forward = _quat_apply_wxyz(quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    forward[2] = 0.0
    return _normalize(forward)


def _box_xy_from_hand_mid(
    hand_mid_xy: np.ndarray,
    forward_xy: np.ndarray,
    forward_extra: float,
) -> np.ndarray:
    """Center laterally between hands; apply forward offset along robot forward only."""
    forward_xy = _normalize(forward_xy.astype(np.float32))
    lateral_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=np.float32)
    forward_coord = float(np.dot(hand_mid_xy, forward_xy)) + forward_extra
    lateral_coord = float(np.dot(hand_mid_xy, lateral_xy))
    return forward_xy * forward_coord + lateral_xy * lateral_coord


def _first_existing(name_to_idx: dict[str, int], candidates: tuple[str, ...]) -> int:
    for name in candidates:
        if name in name_to_idx:
            return name_to_idx[name]
    raise KeyError(f"None of these bodies exist in motion file: {candidates}")


def _detect_pickup_place(body_names: list[str], body_pos_w: np.ndarray) -> tuple[int, int]:
    name_to_idx = {name: idx for idx, name in enumerate(body_names)}

    left_idx = _first_existing(name_to_idx, LEFT_WRIST_CANDIDATES)
    right_idx = _first_existing(name_to_idx, RIGHT_WRIST_CANDIDATES)
    left_pos = body_pos_w[:, left_idx]
    right_pos = body_pos_w[:, right_idx]
    hand_mid = 0.5 * (left_pos + right_pos)

    hand_z = hand_mid[:, 2]
    pickup_frame = int(np.argmin(hand_z))
    pickup_frame = max(0, pickup_frame)

    pelvis_idx = name_to_idx.get("pelvis", 0)
    pelvis_x = body_pos_w[:, pelvis_idx, 0]
    place_frame = int(np.argmin(pelvis_x))

    if place_frame <= pickup_frame + 10:
        pelvis_y = body_pos_w[:, pelvis_idx, 1]
        dy = np.diff(pelvis_y)
        candidate_start = max(pickup_frame + 40, 80)
        place_frame = None
        for frame in range(candidate_start, len(dy) - 4):
            window = dy[frame : frame + 4]
            if np.all(window > 0.0015):
                place_frame = frame
                break
        if place_frame is None:
            place_frame = int(np.argmin(pelvis_y))

    place_frame = max(place_frame, pickup_frame + 20)
    return pickup_frame, place_frame


def _build_trajectory(
    body_names: list[str],
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    pickup_frame: int,
    place_frame: int,
    platform_forward_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_frames = body_pos_w.shape[0]
    name_to_idx = {name: idx for idx, name in enumerate(body_names)}

    left_idx = _first_existing(name_to_idx, LEFT_WRIST_CANDIDATES)
    right_idx = _first_existing(name_to_idx, RIGHT_WRIST_CANDIDATES)
    pelvis_idx = name_to_idx.get("pelvis", 0)
    left_pos = body_pos_w[:, left_idx]
    right_pos = body_pos_w[:, right_idx]
    hand_mid = 0.5 * (left_pos + right_pos)

    pickup_forward_xy = _forward_xy(body_quat_w[pickup_frame, pelvis_idx])[:2]

    ref_pos = body_pos_w[place_frame, pelvis_idx]
    ref_quat = body_quat_w[place_frame, pelvis_idx]
    ref_forward = _forward_xy(ref_quat)

    platform_center = ref_pos.copy()
    platform_center[:2] += ref_forward[:2] * platform_forward_dist

    box_pos = np.zeros((num_frames, 3), dtype=np.float32)
    box_quat = np.tile(OBJECT_REST_QUAT, (num_frames, 1))

    platform_quat = np.zeros((num_frames, 4), dtype=np.float32)
    platform_quat[:, 0] = 1.0

    box_size = np.array([2.0 * s for s in BOX_HALF_XYZ], dtype=np.float32)
    platform_size = np.array([2.0 * s for s in PLATFORM_HALF_XYZ], dtype=np.float32)

    pickup_forward_xy = _forward_xy(body_quat_w[pickup_frame, pelvis_idx])[:2]

    spawn_xy = _box_xy_from_hand_mid(
        hand_mid[pickup_frame, :2], pickup_forward_xy, SPAWN_FORWARD_OFFSET
    )
    spawn_x = float(spawn_xy[0])
    ground_pos = np.array(
        [spawn_x, hand_mid[pickup_frame, 1], BOX_HALF_Z],
        dtype=np.float32,
    )
    z_offset = BOX_HALF_Z - hand_mid[pickup_frame, 2]

    # 1) Before pickup: fixed spawn pose on the ground.
    box_pos[:pickup_frame] = ground_pos

    # 2) From pickup_frame: forward offset in X; Y at the hand midpoint each frame.
    for frame in range(pickup_frame, place_frame + 1):
        box_xy = _box_xy_from_hand_mid(
            hand_mid[frame, :2], pickup_forward_xy, SPAWN_FORWARD_OFFSET
        )
        box_pos[frame, 0] = box_xy[0]
        box_pos[frame, 1] = hand_mid[frame, 1]
        box_pos[frame, 2] = hand_mid[frame, 2] + z_offset

    # 3) After place: freeze at the last carried pose (no snap to platform center).
    if place_frame + 1 < num_frames:
        box_pos[place_frame + 1 :] = box_pos[place_frame]

    box_bottom_z = float(box_pos[place_frame, 2] - BOX_HALF_Z)
    platform_center[2] = box_bottom_z - PLATFORM_HALF_XYZ[2]

    platform_pos = np.zeros((num_frames, 3), dtype=np.float32)
    platform_pos[:] = platform_center

    return box_pos, box_quat, platform_pos, platform_quat, box_size, platform_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lifting-npz",
        required=True,
        help="Path to the robot-only lifting.npz",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the synthetic .npy payload",
    )
    parser.add_argument(
        "--model-xml",
        default=str(G1_SPHERE_HAND_XML),
        help="Robot XML used to resolve body names when the NPZ lacks them",
    )
    parser.add_argument(
        "--pickup-frame",
        type=int,
        default=None,
        help="Frame where the box starts following the hands (default: auto-detect)",
    )
    parser.add_argument(
        "--place-frame",
        type=int,
        default=None,
        help="Last carry frame; box rests on the platform from the next frame (default: auto-detect)",
    )
    parser.add_argument(
        "--platform-forward-dist",
        type=float,
        default=DEFAULT_PLATFORM_FORWARD_DIST,
        help="Place the platform this many meters in front of the robot at place_frame (default: 0.45)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lifting_path = Path(args.lifting_npz)
    output_path = Path(args.output)
    model_xml = Path(args.model_xml)

    motion = _load_motion(lifting_path, model_xml)
    body_names = motion["body_names"]
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]

    if args.pickup_frame is not None and args.place_frame is not None:
        pickup_frame = int(args.pickup_frame)
        place_frame = int(args.place_frame)
    else:
        pickup_frame, place_frame = _detect_pickup_place(body_names, body_pos_w)
        if args.pickup_frame is not None:
            pickup_frame = int(args.pickup_frame)
        if args.place_frame is not None:
            place_frame = int(args.place_frame)

    if not (0 <= pickup_frame < body_pos_w.shape[0]):
        raise ValueError(f"pickup_frame={pickup_frame} out of range [0, {body_pos_w.shape[0]})")
    if not (pickup_frame < place_frame < body_pos_w.shape[0]):
        raise ValueError(
            f"Expected pickup_frame < place_frame < num_frames, got "
            f"{pickup_frame}, {place_frame}, {body_pos_w.shape[0]}"
        )

    (
        box_pos_w,
        box_quat_w,
        platform_pos_w,
        platform_quat_w,
        box_size,
        platform_size,
    ) = _build_trajectory(
        body_names,
        body_pos_w,
        body_quat_w,
        pickup_frame,
        place_frame,
        float(args.platform_forward_dist),
    )

    payload = {
        "fps": motion["fps"],
        "num_frames": int(body_pos_w.shape[0]),
        "pickup_frame": int(pickup_frame),
        "place_frame": int(place_frame),
        "box_pos_w": box_pos_w,
        "box_quat_w": box_quat_w,
        "platform_pos_w": platform_pos_w,
        "platform_quat_w": platform_quat_w,
        "box_size": box_size,
        "platform_size": platform_size,
    }

    np.save(output_path, payload, allow_pickle=True)
    print(f"wrote {output_path}")
    print(f"pickup_frame={pickup_frame}")
    print(f"place_frame={place_frame}")
    print(f"platform_pos={platform_pos_w[0]}")
    print(f"platform_quat(wxyz)={platform_quat_w[0]}")
    print(f"box_quat pickup={box_quat_w[pickup_frame]}")
    print(f"platform_size={platform_size}")


if __name__ == "__main__":
    main()
