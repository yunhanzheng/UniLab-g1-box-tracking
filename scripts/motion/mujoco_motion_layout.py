"""Expand compact body motion arrays to MuJoCo body-id layout."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

_BODY_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _expand_from_compact(
    body_arrays: dict[str, np.ndarray],
    body_names: Sequence[str],
    model: mujoco.MjModel,
) -> dict[str, np.ndarray]:
    name_to_src = {str(name): idx for idx, name in enumerate(body_names)}
    num_bodies = int(model.nbody)

    expanded: dict[str, np.ndarray] = {}
    for key in _BODY_KEYS:
        if key not in body_arrays:
            continue
        src = np.asarray(body_arrays[key], dtype=np.float32)
        if src.ndim != 3:
            raise ValueError(f"Expected {key} with shape (T, B, C), got {src.shape}")
        out = np.zeros((src.shape[0], num_bodies, src.shape[2]), dtype=np.float32)
        for body_id in range(num_bodies):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not body_name:
                continue
            src_idx = name_to_src.get(body_name)
            if src_idx is None:
                continue
            out[:, body_id] = src[:, src_idx]
        expanded[key] = out
    return expanded


def _expand_from_body_id_layout(
    body_arrays: dict[str, np.ndarray],
    src_model: mujoco.MjModel,
    dst_model: mujoco.MjModel,
) -> dict[str, np.ndarray]:
    num_bodies = int(dst_model.nbody)
    expanded: dict[str, np.ndarray] = {}
    for key in _BODY_KEYS:
        if key not in body_arrays:
            continue
        src = np.asarray(body_arrays[key], dtype=np.float32)
        if src.ndim != 3:
            raise ValueError(f"Expected {key} with shape (T, B, C), got {src.shape}")
        if src.shape[1] != int(src_model.nbody):
            raise ValueError(
                f"Expected {key} body axis to match source model nbody "
                f"({src_model.nbody}), got {src.shape[1]}"
            )
        out = np.zeros((src.shape[0], num_bodies, src.shape[2]), dtype=np.float32)
        for dst_body_id in range(num_bodies):
            body_name = mujoco.mj_id2name(dst_model, mujoco.mjtObj.mjOBJ_BODY, dst_body_id)
            if not body_name:
                continue
            src_body_id = mujoco.mj_name2id(src_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if src_body_id < 0:
                continue
            out[:, dst_body_id] = src[:, src_body_id]
        expanded[key] = out
    return expanded


def expand_body_arrays_to_mujoco_body_ids(
    body_arrays: dict[str, np.ndarray],
    body_names: Sequence[str],
    model_xml: str | Path,
    source_model_xml: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Reindex body arrays to MuJoCo ``body_id`` slots for ``model_xml``.

    Supports two source layouts:

    * **compact** — ``body_arrays[*].shape[1] == len(body_names)`` and columns follow
      ``body_names`` order.
    * **body-id** — columns already follow another model's MuJoCo body ids. Pass
      ``source_model_xml`` for that model (required when ``len(body_names)`` differs
      from the source body axis).
    """
    dst_model = mujoco.MjModel.from_xml_path(str(model_xml))
    sample_key = next(key for key in _BODY_KEYS if key in body_arrays)
    src_width = int(np.asarray(body_arrays[sample_key]).shape[1])
    dst_width = int(dst_model.nbody)

    if src_width == dst_width:
        return {
            key: np.asarray(body_arrays[key], dtype=np.float32).copy()
            for key in _BODY_KEYS
            if key in body_arrays
        }

    if src_width == len(body_names):
        return _expand_from_compact(body_arrays, body_names, dst_model)

    if source_model_xml is None:
        raise ValueError(
            f"Body axis width {src_width} does not match compact body_names "
            f"({len(body_names)}) or destination nbody ({dst_width}). "
            "Pass source_model_xml when the input is already body-id layout."
        )

    src_model = mujoco.MjModel.from_xml_path(str(source_model_xml))
    return _expand_from_body_id_layout(body_arrays, src_model, dst_model)
