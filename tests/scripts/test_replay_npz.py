"""Tests for NPZ motion replay helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from scripts.motion.replay_npz import _resolve_root_body_index, _uses_body_id_layout

ROOT = Path(__file__).resolve().parents[2]
LIFTING_BOX_NPZ = ROOT / "scripts" / "motion" / "lifting_unilab_box.npz"
LIFTING_NPZ = ROOT / "scripts" / "motion" / "lifting_unilab.npz"
SCENE_XML = ROOT / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat_with_largebox.xml"
REPLAY_MOTRIX = ROOT / "scripts" / "motion" / "replay_npz_motrix.py"


def test_body_id_layout_detected_for_unilab_lifting_npz():
    data = np.load(LIFTING_BOX_NPZ, allow_pickle=True)
    body_names = [str(x) for x in data["body_names"]]
    body_pos_w = data["body_pos_w"]

    assert _uses_body_id_layout(body_names, body_pos_w)
    assert len(body_names) < body_pos_w.shape[1]


def test_resolve_root_body_index_uses_mujoco_pelvis_body_id():
    data = np.load(LIFTING_BOX_NPZ, allow_pickle=True)
    body_names = [str(x) for x in data["body_names"]]
    body_pos_w = data["body_pos_w"]
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    root_id = _resolve_root_body_index(body_names, body_pos_w, model=model)

    assert pelvis_id == 1
    assert root_id == pelvis_id
    assert body_names.index("pelvis") == 0
    assert np.allclose(body_pos_w[55, 0], 0.0)
    assert body_pos_w[55, root_id, 2] > 0.2


def test_replay_npz_motrix_dry_run_box_and_flat():
    if not LIFTING_BOX_NPZ.is_file():
        import pytest

        pytest.skip("lifting_unilab_box.npz fixture not present")
    for npz in (LIFTING_BOX_NPZ, LIFTING_NPZ):
        if not npz.is_file():
            continue
        proc = subprocess.run(
            [
                sys.executable,
                str(REPLAY_MOTRIX),
                "--npz_file",
                str(npz),
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Dry run OK" in proc.stdout
