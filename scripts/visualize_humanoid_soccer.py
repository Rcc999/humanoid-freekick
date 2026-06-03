#!/usr/bin/env python3
"""
Visualize HumanoidSoccer .npz motion files using ASAP's G1 MuJoCo model.

Usage (spark - inside VNC):
    cd ~/rayane/humanoid-freekick
    source venv_asap/bin/activate
    python scripts/visualize_humanoid_soccer.py \
        ~/rayane/HumanoidSoccer/motions/soccer-stylized/football_stylized-001_right.npz

Controls:
    Space  — pause / resume
    R      — reset to frame 0
    N      — next file (if multiple passed)
"""

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

ROBOT_XML = Path(__file__).parent.parent / \
    "third_party/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_fitmotionONLY.xml"


def load_motion(path):
    d = np.load(path, allow_pickle=True)
    fps      = int(d['fps'][0])
    joint_pos = d['joint_pos']          # (T, 29)
    root_pos  = d['body_pos_w'][:, 0]   # (T, 3)  — pelvis world position
    root_quat = d['body_quat_w'][:, 0]  # (T, 4)  — pelvis quaternion (w,x,y,z)
    kick_leg  = str(d['kick_leg'])
    T = joint_pos.shape[0]
    print(f"  {Path(path).name}: {T} frames @ {fps}fps = {T/fps:.1f}s  |  kick leg: {kick_leg}")
    return fps, root_pos, root_quat, joint_pos


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python visualize_humanoid_soccer.py file1.npz [file2.npz ...]")
        sys.exit(1)

    print(f"Robot XML: {ROBOT_XML}")
    mj_model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    mj_data  = mujoco.MjData(mj_model)

    file_idx   = 0
    frame      = 0
    paused     = False
    last_time  = time.time()

    fps, root_pos, root_quat, joint_pos = load_motion(paths[file_idx])
    dt = 1.0 / fps

    def key_callback(keycode):
        nonlocal frame, paused, file_idx, fps, root_pos, root_quat, joint_pos, dt
        ch = chr(keycode)
        if ch == ' ':
            paused = not paused
            print("Paused" if paused else "Playing")
        elif ch == 'R':
            frame = 0
            print("Reset")
        elif ch == 'N':
            file_idx = (file_idx + 1) % len(paths)
            fps, root_pos, root_quat, joint_pos = load_motion(paths[file_idx])
            dt = 1.0 / fps
            frame = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data,
                                      key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            T = joint_pos.shape[0]
            t = frame % T

            # Set root position
            mj_data.qpos[0:3] = root_pos[t]

            # Set root quaternion — HumanoidSoccer stores (w,x,y,z), MuJoCo expects (w,x,y,z)
            mj_data.qpos[3:7] = root_quat[t]

            # Set joint positions (29 DOF)
            n_joints = min(joint_pos.shape[1], mj_model.nq - 7)
            mj_data.qpos[7:7 + n_joints] = joint_pos[t, :n_joints]

            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            if not paused:
                frame += 1

            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
