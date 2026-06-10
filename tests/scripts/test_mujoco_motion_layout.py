from __future__ import annotations

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH


def test_expand_body_arrays_to_mujoco_body_ids_supports_motrix_motion_indices():
    mujoco = pytest.importorskip("mujoco")
    from scripts.motion.mujoco_motion_layout import expand_body_arrays_to_mujoco_body_ids

    body_names = ["pelvis", "left_knee_link", "right_wrist_yaw_link"]
    src = np.arange(2 * len(body_names) * 3, dtype=np.float32).reshape(2, len(body_names), 3)
    expanded = expand_body_arrays_to_mujoco_body_ids(
        {"body_pos_w": src},
        body_names,
        ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_with_largebox.xml",
    )["body_pos_w"]

    model = mujoco.MjModel.from_xml_path(
        str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_with_largebox.xml")
    )
    assert expanded.shape[1] == model.nbody

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    np.testing.assert_allclose(expanded[:, pelvis_id], src[:, 0])
    np.testing.assert_allclose(expanded[:, wrist_id], src[:, 2])


def test_expand_body_id_layout_preserves_pelvis_when_growing_scene():
    mujoco = pytest.importorskip("mujoco")
    from scripts.motion.mujoco_motion_layout import expand_body_arrays_to_mujoco_body_ids

    robot_xml = ASSETS_ROOT_PATH / "robots" / "g1" / "g1_sphere_hand.xml"
    scene_xml = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat_with_largebox.xml"
    robot_model = mujoco.MjModel.from_xml_path(str(robot_xml))
    pelvis_id = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    src_quat = np.zeros((3, robot_model.nbody, 4), dtype=np.float32)
    src_quat[:, pelvis_id] = np.array([0.0, 0.1, 0.0, 0.995], dtype=np.float32)
    body_names = ["pelvis"]  # intentionally mismatched length; body-id layout is inferred

    expanded = expand_body_arrays_to_mujoco_body_ids(
        {"body_quat_w": src_quat},
        body_names,
        scene_xml,
        source_model_xml=robot_xml,
    )["body_quat_w"]

    scene_model = mujoco.MjModel.from_xml_path(str(scene_xml))
    scene_pelvis_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    np.testing.assert_allclose(expanded[:, scene_pelvis_id], src_quat[:, pelvis_id])
