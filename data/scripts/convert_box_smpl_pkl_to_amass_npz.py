#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
#
# Convert custom SMPL PKL motions to AMASS-style NPZ expected by
# data/scripts/convert_amass_to_proto.py.
#
# Input PKL format (from GMR parse_mvnx_simple.py):
#   - pose:  (T, 24, 3, 3) local joint rotation matrices
#   - transl:(T, 3) root translation in meters
#
# Output NPZ format:
#   - poses:            (T, 72) axis-angle (SMPL 24 joints * 3)
#   - trans:            (T, 3)
#   - mocap_framerate:  scalar

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot


DEFAULT_INPUT_DIR = "/home/yk/Desktop/GMR/human_motion_data_new/box_smpl_data"
DEFAULT_OUTPUT_DIR = "/home/yk/Desktop/ProtoMotions/data/box"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except TypeError:
            return pickle.load(f, encoding="latin1")


def validate_sample(data: dict, file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if "pose" not in data or "transl" not in data:
        raise KeyError(
            f"{file_path}: missing keys. Expected ['pose', 'transl'], got {list(data.keys())}"
        )

    pose = data["pose"]
    transl = data["transl"]

    if not isinstance(pose, np.ndarray) or not isinstance(transl, np.ndarray):
        raise TypeError(f"{file_path}: 'pose' and 'transl' must both be numpy arrays")

    if pose.ndim != 4 or pose.shape[1:] != (24, 3, 3):
        raise ValueError(
            f"{file_path}: 'pose' must be (T,24,3,3), got {pose.shape}"
        )

    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"{file_path}: 'transl' must be (T,3), got {transl.shape}")

    if pose.shape[0] != transl.shape[0]:
        raise ValueError(
            f"{file_path}: frame mismatch pose={pose.shape[0]} transl={transl.shape[0]}"
        )

    if pose.shape[0] < 2:
        raise ValueError(f"{file_path}: too few frames ({pose.shape[0]}), need >= 2")

    return pose, transl


def convert_pose_mats_to_axis_angle(pose_mats: np.ndarray) -> np.ndarray:
    # (T,24,3,3) -> (T*24,3,3) -> rotvec -> (T,24,3) -> (T,72)
    t = pose_mats.shape[0]
    rotvec = sRot.from_matrix(pose_mats.reshape(-1, 3, 3)).as_rotvec()
    return rotvec.reshape(t, 24, 3).reshape(t, 72).astype(np.float32)


def apply_world_frame_transform(
    poses: np.ndarray,
    trans: np.ndarray,
    world_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply optional world-frame conversion to match downstream AMASS-style assumptions.

    The `yup_to_zup` mode matches the transform used in the user's GMR pipeline:
    - translation: (x, y, z) -> (z, x, y)
    - root orientation: R_root -> C @ R_root, where
      C = [[0,0,1],[1,0,0],[0,1,0]]
    """
    if world_frame == "none":
        return poses, trans
    if world_frame != "yup_to_zup":
        raise ValueError(f"Unsupported world_frame: {world_frame}")

    poses_out = poses.copy()
    trans_out = np.zeros_like(trans)
    trans_out[:, 0] = trans[:, 2]
    trans_out[:, 1] = trans[:, 0]
    trans_out[:, 2] = trans[:, 1]

    corr_rotmat_sf = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    root_orient = poses_out[:, :3].astype(np.float64)
    root_rotmat = sRot.from_rotvec(root_orient).as_matrix()
    root_rotmat_converted = np.einsum("ij,tjk->tik", corr_rotmat_sf, root_rotmat)
    poses_out[:, :3] = sRot.from_matrix(root_rotmat_converted).as_rotvec().astype(
        np.float32
    )

    return poses_out, trans_out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert custom SMPL PKL files to AMASS-style NPZ."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help="Directory containing source .pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory to write AMASS-style .npz files",
    )
    parser.add_argument(
        "--mocap-fps",
        type=float,
        default=240.0,
        help="Value written to mocap_framerate in output NPZ",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input-dir for .pkl files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion when target .npz already exists",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Convert at most N files (for quick tests)",
    )
    parser.add_argument(
        "--world-frame",
        type=str,
        choices=["yup_to_zup", "none"],
        default="yup_to_zup",
        help=(
            "World-frame conversion applied to root/trans before saving NPZ. "
            "'yup_to_zup' matches the GMR conversion chain; 'none' disables it."
        ),
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.pkl" if args.recursive else "*.pkl"
    files = sorted(args.input_dir.glob(pattern))
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        raise RuntimeError(f"No .pkl files found in {args.input_dir} (recursive={args.recursive})")

    ok = 0
    skipped = 0
    failed = 0

    for src in files:
        rel = src.relative_to(args.input_dir)
        dst = (args.output_dir / rel).with_suffix(".npz")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and dst.exists():
            skipped += 1
            continue

        try:
            data = load_pickle(src)
            pose_mats, transl = validate_sample(data, src)
            poses = convert_pose_mats_to_axis_angle(pose_mats.astype(np.float64))
            trans = transl.astype(np.float32)
            poses, trans = apply_world_frame_transform(
                poses=poses,
                trans=trans,
                world_frame=args.world_frame,
            )

            np.savez_compressed(
                dst,
                poses=poses,
                trans=trans,
                mocap_framerate=np.array(args.mocap_fps, dtype=np.float32),
            )
            ok += 1
            print(f"[OK] {src.name} -> {dst}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {src}: {e}")

    print("\n=== Summary ===")
    print(f"input_dir   : {args.input_dir}")
    print(f"output_dir  : {args.output_dir}")
    print(f"converted   : {ok}")
    print(f"skipped     : {skipped}")
    print(f"failed      : {failed}")
    print(f"total_seen  : {len(files)}")

    if failed > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
