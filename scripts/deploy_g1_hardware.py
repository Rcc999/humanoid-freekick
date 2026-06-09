"""
Hardware deployment of the CR7 free-kick policy on the Unitree G1 (29-DOF).

Mirrors the structure of scripts/sim2sim_mujoco.py but reads sensors from the
real robot (Unitree SDK2) and sends motor commands instead of stepping MuJoCo.

────────────────────────────────────────────────────────────────────────────
SAFETY — READ BEFORE RUNNING

This script will command motors on a real humanoid robot. A fall can damage
the hardware and injure people nearby. Before running:

  1. Robot should be on a safety harness (gantry/rope) for the first runs.
  2. Have a hardware E-stop within arm's reach.
  3. Run with --suspended first to verify telemetry without floor contact.
  4. Run with --dry_run to verify obs and inference work without sending
     any motor commands at all.
  5. Only then run a full deployment with --execute.

Tested against unitree_sdk2_python 1.x. The G1 must be in "developer mode"
(L1+R1 then L1+A on the controller, or via the web UI) so the high-level
sport_client is disabled and only your LowCmd commands move the motors.
────────────────────────────────────────────────────────────────────────────

Typical usage (policy auto-downloads from Hugging Face on first run):

    # Stage 1 — dry run: read sensors, run policy, do NOT command motors.
    python deploy_g1_hardware.py --network_interface eth0 --dry_run

    # Stage 2 — suspended on harness, no floor contact, no ball.
    python deploy_g1_hardware.py --network_interface eth0 --suspended --execute

    # Stage 3 — full deployment with ball at the motion's natural endpoint.
    python deploy_g1_hardware.py --network_interface eth0 \\
        --ball_xy 0.06,-1.45 --execute

    # Optional — use a local ONNX instead of downloading from HF.
    python deploy_g1_hardware.py --network_interface eth0 \\
        --onnx ./policy_12000_dr_cr7.onnx --dry_run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort

# Default policy on Hugging Face — the 12k DR-fine-tuned checkpoint that
# passed sim2sim + noise validation. Pulled automatically unless --onnx is
# given as a local path.
DEFAULT_HF_REPO = "Shish999/cr7-freekick-g1"
DEFAULT_HF_FILE = "dr/2026-06-04_17-19-06/exported/policy_12000_football_stylized-001_right.onnx"

# ── Unitree SDK2 imports ──────────────────────────────────────────────────
# pip install unitree_sdk2_python   (or build from source on the robot)
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__LowCmd_,
        unitree_hg_msg_dds__LowState_,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
except ImportError as e:
    sys.exit(
        "ERROR: unitree_sdk2_python is required.\n"
        "       Install via: pip install unitree_sdk2_python\n"
        "       Or build it on the robot: https://github.com/unitreerobotics/unitree_sdk2_python\n"
        f"       Original error: {e}"
    )

# ──────────────────────────────────────────────────────────────────────────
# Constants matching IsaacLab/MuJoCo conventions
# ──────────────────────────────────────────────────────────────────────────

# G1 29-DOF joint order as exposed by the Unitree SDK LowState.motor_state[]
# array. This MUST match the firmware on your robot — verify by printing the
# joint names if your SDK provides a mapping, or by toggling one joint at a
# time. Source: unitree_rl_lab/deploy/robots/g1_29dof/.
G1_SDK_JOINT_ORDER = [
    "left_hip_pitch_joint",      #  0
    "left_hip_roll_joint",       #  1
    "left_hip_yaw_joint",        #  2
    "left_knee_joint",           #  3
    "left_ankle_pitch_joint",    #  4
    "left_ankle_roll_joint",     #  5
    "right_hip_pitch_joint",     #  6
    "right_hip_roll_joint",      #  7
    "right_hip_yaw_joint",       #  8
    "right_knee_joint",          #  9
    "right_ankle_pitch_joint",   # 10
    "right_ankle_roll_joint",    # 11
    "waist_yaw_joint",           # 12
    "waist_roll_joint",          # 13
    "waist_pitch_joint",         # 14
    "left_shoulder_pitch_joint", # 15
    "left_shoulder_roll_joint",  # 16
    "left_shoulder_yaw_joint",   # 17
    "left_elbow_joint",          # 18
    "left_wrist_roll_joint",     # 19
    "left_wrist_pitch_joint",    # 20
    "left_wrist_yaw_joint",      # 21
    "right_shoulder_pitch_joint",# 22
    "right_shoulder_roll_joint", # 23
    "right_shoulder_yaw_joint",  # 24
    "right_elbow_joint",         # 25
    "right_wrist_roll_joint",    # 26
    "right_wrist_pitch_joint",   # 27
    "right_wrist_yaw_joint",     # 28
]
NUM_MOTORS = len(G1_SDK_JOINT_ORDER)  # 29

CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

# G1 joint position limits (radians) — extracted from the official MJCF
# in third_party/HumanoidSoccer/.../mjcf/g1.xml. Used to clip target_joint_pos
# before PD so the policy can't drive the motor into a hard joint stop.
# Order matches G1_SDK_JOINT_ORDER above (index 0 = LeftHipPitch).
G1_JOINT_LIMITS_SDK = np.array([
    # legs
    (-2.5307,  2.8798),   # 0  left_hip_pitch
    (-0.5236,  2.9671),   # 1  left_hip_roll
    (-2.7576,  2.7576),   # 2  left_hip_yaw
    (-0.0873,  2.8798),   # 3  left_knee
    (-0.8727,  0.5236),   # 4  left_ankle_pitch
    (-0.2618,  0.2618),   # 5  left_ankle_roll
    (-2.5307,  2.8798),   # 6  right_hip_pitch
    (-2.9671,  0.5236),   # 7  right_hip_roll
    (-2.7576,  2.7576),   # 8  right_hip_yaw
    (-0.0873,  2.8798),   # 9  right_knee
    (-0.8727,  0.5236),   # 10 right_ankle_pitch
    (-0.2618,  0.2618),   # 11 right_ankle_roll
    # waist
    (-2.6180,  2.6180),   # 12 waist_yaw
    (-0.5200,  0.5200),   # 13 waist_roll
    (-0.5200,  0.5200),   # 14 waist_pitch
    # left arm
    (-3.0892,  2.6704),   # 15 left_shoulder_pitch
    (-1.5882,  2.2515),   # 16 left_shoulder_roll
    (-2.6180,  2.6180),   # 17 left_shoulder_yaw
    (-1.0472,  2.0944),   # 18 left_elbow
    (-1.9722,  1.9722),   # 19 left_wrist_roll
    (-1.6144,  1.6144),   # 20 left_wrist_pitch
    (-1.6144,  1.6144),   # 21 left_wrist_yaw
    # right arm
    (-3.0892,  2.6704),   # 22 right_shoulder_pitch
    (-2.2515,  1.5882),   # 23 right_shoulder_roll
    (-2.6180,  2.6180),   # 24 right_shoulder_yaw
    (-1.0472,  2.0944),   # 25 right_elbow
    (-1.9722,  1.9722),   # 26 right_wrist_roll
    (-1.6144,  1.6144),   # 27 right_wrist_pitch
    (-1.6144,  1.6144),   # 28 right_wrist_yaw
], dtype=np.float32)
G1_JOINT_MIN_SDK = G1_JOINT_LIMITS_SDK[:, 0]
G1_JOINT_MAX_SDK = G1_JOINT_LIMITS_SDK[:, 1]

# Motor command mode for G1 PMSM motors (position/torque hybrid).
# Set to 0x01 for normal operation; 0x00 disables the motor.
MOTOR_MODE_ENABLED = 0x01
MOTOR_MODE_DISABLED = 0x00

# Topic names for the unitree_hg (G1) message type.
LOWSTATE_TOPIC = "rt/lowstate"
LOWCMD_TOPIC = "rt/lowcmd"


# ──────────────────────────────────────────────────────────────────────────
# Quaternion helpers (same conventions as sim2sim_mujoco.py — [w, x, y, z])
# ──────────────────────────────────────────────────────────────────────────

def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul(quat_mul(q, qv), quat_inv(q))[1:]


# ──────────────────────────────────────────────────────────────────────────
# State observer — wraps the Unitree LowState subscription
# ──────────────────────────────────────────────────────────────────────────

class G1StateObserver:
    """Subscribes to LowState and exposes the latest sensor reading."""

    def __init__(self):
        self._latest: LowState_ | None = None
        self._sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self._sub.Init(self._callback, 10)

    def _callback(self, msg: LowState_):
        self._latest = msg

    def wait_for_first_message(self, timeout_s: float = 5.0):
        t0 = time.time()
        while self._latest is None:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(
                    f"No LowState received within {timeout_s}s. "
                    "Check that the robot is powered on, in dev mode, and the "
                    "network_interface is correct."
                )
            time.sleep(0.05)

    @property
    def state(self) -> LowState_:
        assert self._latest is not None, "Call wait_for_first_message() first."
        return self._latest

    # ── Convenience accessors ────────────────────────────────────────────
    def base_quat_wxyz(self) -> np.ndarray:
        """Pelvis orientation quaternion in [w, x, y, z]."""
        q = self.state.imu_state.quaternion  # SDK returns [w, x, y, z]
        return np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)

    def base_ang_vel_body(self) -> np.ndarray:
        """Angular velocity in BODY frame (IMU gyroscope reading)."""
        # The G1 IMU reports angular velocity already in the base body frame.
        # This matches MuJoCo's free-joint qvel[3:6] convention — DO NOT
        # double-rotate.
        gyro = self.state.imu_state.gyroscope
        return np.array([gyro[0], gyro[1], gyro[2]], dtype=np.float32)

    def joint_pos_sdk_order(self) -> np.ndarray:
        return np.array(
            [self.state.motor_state[i].q for i in range(NUM_MOTORS)],
            dtype=np.float32,
        )

    def joint_vel_sdk_order(self) -> np.ndarray:
        return np.array(
            [self.state.motor_state[i].dq for i in range(NUM_MOTORS)],
            dtype=np.float32,
        )


# ──────────────────────────────────────────────────────────────────────────
# Motor command publisher — wraps LowCmd
# ──────────────────────────────────────────────────────────────────────────

class G1MotorPublisher:
    def __init__(self):
        self._pub = ChannelPublisher(LOWCMD_TOPIC, LowCmd_)
        self._pub.Init()
        self._crc = CRC()
        self._cmd: LowCmd_ = unitree_hg_msg_dds__LowCmd_()
        # The G1 hg LowCmd has a `mode_pr` and `mode_machine` field. Set them
        # to the values used by the demo controller (0 / 0 is safe defaults).
        self._cmd.mode_pr = 0
        self._cmd.mode_machine = 0
        # Disable every motor by default until set_torque is called.
        for i in range(NUM_MOTORS):
            self._cmd.motor_cmd[i].mode = MOTOR_MODE_DISABLED
            self._cmd.motor_cmd[i].q = 0.0
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kp = 0.0
            self._cmd.motor_cmd[i].kd = 0.0
            self._cmd.motor_cmd[i].tau = 0.0

    def send_torque(self, tau_sdk_order: np.ndarray):
        """Send pure-torque commands (kp=kd=0). One vector entry per motor."""
        assert tau_sdk_order.shape == (NUM_MOTORS,)
        for i in range(NUM_MOTORS):
            self._cmd.motor_cmd[i].mode = MOTOR_MODE_ENABLED
            self._cmd.motor_cmd[i].q = 0.0
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kp = 0.0
            self._cmd.motor_cmd[i].kd = 0.0
            self._cmd.motor_cmd[i].tau = float(tau_sdk_order[i])
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def send_position(
        self,
        q_sdk_order: np.ndarray,
        kp_sdk_order: np.ndarray,
        kd_sdk_order: np.ndarray,
    ):
        """Send position targets with onboard PD. Used during warm-up."""
        for i in range(NUM_MOTORS):
            self._cmd.motor_cmd[i].mode = MOTOR_MODE_ENABLED
            self._cmd.motor_cmd[i].q = float(q_sdk_order[i])
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kp = float(kp_sdk_order[i])
            self._cmd.motor_cmd[i].kd = float(kd_sdk_order[i])
            self._cmd.motor_cmd[i].tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def damping_all(self, kd_damp: float = 5.0):
        """Soft stop: motors stay enabled but only apply damping (no position
        hold, no torque target). The robot collapses gently as gravity pulls
        it down, instead of slamming joints into stops as with hard disable.

        This matches what the Unitree controller's L2+B does. Always prefer
        this for safety-driven aborts during dynamic motion.
        """
        for i in range(NUM_MOTORS):
            self._cmd.motor_cmd[i].mode = MOTOR_MODE_ENABLED
            self._cmd.motor_cmd[i].q = 0.0
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kp = 0.0
            self._cmd.motor_cmd[i].kd = float(kd_damp)
            self._cmd.motor_cmd[i].tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def disable_all(self):
        """Hard disable: motors fully released, no torque, no damping. Use
        only at end-of-session or when you really want the robot to go
        completely limp. For abort-during-motion, prefer damping_all()."""
        for i in range(NUM_MOTORS):
            self._cmd.motor_cmd[i].mode = MOTOR_MODE_DISABLED
            self._cmd.motor_cmd[i].q = 0.0
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kp = 0.0
            self._cmd.motor_cmd[i].kd = 0.0
            self._cmd.motor_cmd[i].tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)


# ──────────────────────────────────────────────────────────────────────────
# Policy loader and motion table
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyAssets:
    sess: ort.InferenceSession
    joint_names_policy: list[str]
    joint_stiffness: np.ndarray   # policy order
    joint_damping: np.ndarray     # policy order
    default_joint_pos: np.ndarray # policy order
    action_scale: np.ndarray      # policy order
    anchor_body_name: str
    body_names: list[str]
    policy_to_sdk: np.ndarray  # index map: policy_idx -> sdk_idx
    sdk_to_policy: np.ndarray  # index map: sdk_idx -> policy_idx
    torque_limits_policy: np.ndarray  # per-joint effort limits in policy order


def resolve_policy_path(
    local_path: str | None,
    hf_repo: str,
    hf_file: str,
    cache_dir: str | None = None,
) -> str:
    """Return a local filesystem path to the policy ONNX.

    Priority:
      1. `local_path` if given and exists → use as-is.
      2. Otherwise download `hf_file` from `hf_repo` on Hugging Face.

    Both `.onnx` and the (sometimes-present) `.onnx.data` external-weights
    file are downloaded so ONNX Runtime can load the model correctly.
    """
    if local_path and os.path.exists(local_path):
        print(f"[deploy] Using local ONNX: {local_path}")
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "ERROR: `huggingface_hub` is required to fetch the policy.\n"
            "       Install via: pip install huggingface_hub\n"
            "       Or pass a local --onnx path."
        )

    print(f"[deploy] Downloading policy from HF: {hf_repo}/{hf_file}")
    onnx_path = hf_hub_download(
        repo_id=hf_repo, filename=hf_file, cache_dir=cache_dir
    )
    # Also fetch the external-weights sidecar if it exists. ONNX Runtime
    # auto-loads `<name>.onnx.data` if it sits next to the .onnx file.
    try:
        hf_hub_download(
            repo_id=hf_repo, filename=hf_file + ".data", cache_dir=cache_dir
        )
    except Exception:
        pass  # No sidecar needed for small models
    print(f"[deploy] Policy cached at: {onnx_path}")
    return onnx_path


def load_policy(onnx_path: str) -> PolicyAssets:
    print(f"[deploy] Loading ONNX from {onnx_path}")
    model = onnx.load(onnx_path)
    meta = {p.key: p.value for p in model.metadata_props}

    joint_names = meta["joint_names"].split(",")
    kp = np.array(meta["joint_stiffness"].split(","), dtype=np.float32)
    kd = np.array(meta["joint_damping"].split(","), dtype=np.float32)
    default_q = np.array(meta["default_joint_pos"].split(","), dtype=np.float32)
    action_scale = np.array(meta["action_scale"].split(","), dtype=np.float32)
    anchor = meta["anchor_body_name"]
    body_names = meta["body_names"].split(",")

    assert len(joint_names) == NUM_MOTORS, (
        f"Policy expects {len(joint_names)} joints but G1 SDK exposes "
        f"{NUM_MOTORS}. Did you export the right policy?"
    )

    # Build the policy ↔ SDK joint index maps.
    policy_to_sdk = np.array(
        [G1_SDK_JOINT_ORDER.index(name) for name in joint_names], dtype=np.int32
    )
    sdk_to_policy = np.argsort(policy_to_sdk).astype(np.int32)

    # Per-joint effort limits in IsaacSim training (effort_limit_sim). These
    # are the same values used in sim2sim_mujoco.py for torque clipping.
    LIMITS_BY_REGEX = {
        ".*hip_yaw.*": 88.0,
        ".*hip_roll.*": 139.0,
        ".*hip_pitch.*": 88.0,
        ".*knee.*": 139.0,
        ".*ankle.*": 50.0,
        ".*waist_yaw.*": 88.0,
        ".*waist_roll.*": 50.0,
        ".*waist_pitch.*": 50.0,
        ".*shoulder.*": 25.0,
        ".*elbow.*": 25.0,
        ".*wrist_roll.*": 25.0,
        ".*wrist_pitch.*": 5.0,
        ".*wrist_yaw.*": 5.0,
    }
    import re
    torque_limits = np.zeros(NUM_MOTORS, dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, lim in LIMITS_BY_REGEX.items():
            if re.fullmatch(pattern, name):
                torque_limits[i] = lim
                break
        else:
            print(f"[deploy] WARNING: no effort limit found for {name}, using 50 Nm")
            torque_limits[i] = 50.0

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    return PolicyAssets(
        sess=sess,
        joint_names_policy=joint_names,
        joint_stiffness=kp,
        joint_damping=kd,
        default_joint_pos=default_q,
        action_scale=action_scale,
        anchor_body_name=anchor,
        body_names=body_names,
        policy_to_sdk=policy_to_sdk,
        sdk_to_policy=sdk_to_policy,
        torque_limits_policy=torque_limits,
    )


def precompute_motion_table(assets: PolicyAssets):
    """Pre-evaluate the ONNX with each time_step to extract the motion
    reference at every frame. Avoids the 1-step lag bug from sim2sim."""
    dummy_obs = np.zeros((1, 160), dtype=np.float32)
    h0 = np.zeros((2, 1, 128), dtype=np.float32)
    c0 = np.zeros((2, 1, 128), dtype=np.float32)
    out0 = assets.sess.run(
        None,
        {"obs": dummy_obs, "h_in": h0, "c_in": c0, "time_step": np.zeros((1, 1), dtype=np.float32)},
    )
    total = int(out0[-1][0, 0])
    print(f"[deploy] Motion has {total} frames ({total / CONTROL_HZ:.2f}s @ {CONTROL_HZ:.0f}Hz)")

    anchor_idx = assets.body_names.index(assets.anchor_body_name)
    joint_pos_table = np.zeros((total, NUM_MOTORS), dtype=np.float32)
    joint_vel_table = np.zeros((total, NUM_MOTORS), dtype=np.float32)
    anchor_av_table = np.zeros((total, 3), dtype=np.float32)
    anchor_pos_table = np.zeros((total, 3), dtype=np.float32)
    for t in range(total):
        out = assets.sess.run(
            None,
            {
                "obs": dummy_obs,
                "h_in": h0,
                "c_in": c0,
                "time_step": np.array([[t]], dtype=np.float32),
            },
        )
        joint_pos_table[t] = out[3][0]
        joint_vel_table[t] = out[4][0]
        anchor_pos_table[t] = out[5][0][anchor_idx, :]
        anchor_av_table[t] = out[8][0][anchor_idx, :]
    return {
        "joint_pos": joint_pos_table,
        "joint_vel": joint_vel_table,
        "anchor_av": anchor_av_table,
        "anchor_pos": anchor_pos_table,
        "total": total,
        "anchor_idx": anchor_idx,
    }


# ──────────────────────────────────────────────────────────────────────────
# Observation construction
# ──────────────────────────────────────────────────────────────────────────

def projected_gravity_body(base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity vector projected into the base body frame."""
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return quat_rotate(quat_inv(base_quat_wxyz), g_world)


def world_point_in_pelvis_frame(
    point_world: np.ndarray,
    pelvis_world: np.ndarray,
    pelvis_quat_wxyz: np.ndarray,
) -> np.ndarray:
    delta = point_world - pelvis_world
    return quat_rotate(quat_inv(pelvis_quat_wxyz), delta)


# ──────────────────────────────────────────────────────────────────────────
# Main deployment loop
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=None,
                        help="Path to a LOCAL exported policy ONNX. If omitted, "
                             f"the script downloads from --hf_repo / --hf_file "
                             f"(default: {DEFAULT_HF_REPO}/{DEFAULT_HF_FILE}).")
    parser.add_argument("--hf_repo", default=DEFAULT_HF_REPO,
                        help="Hugging Face repo to pull the policy from when "
                             "no local --onnx is given.")
    parser.add_argument("--hf_file", default=DEFAULT_HF_FILE,
                        help="Path inside the HF repo to the policy ONNX.")
    parser.add_argument("--hf_cache_dir", default=None,
                        help="Cache directory for HF downloads. Defaults to "
                             "~/.cache/huggingface/hub.")
    parser.add_argument("--network_interface", required=True,
                        help="Network interface connected to the G1 (e.g. eth0)")
    parser.add_argument("--ball_xy", type=str, default="0.06,-1.45",
                        help='Ball position in the robot-local world frame as "x,y" '
                             '(meters). Default matches the motion file last anchor.')
    parser.add_argument("--destination_xy", type=str, default=None,
                        help='Optional override for target_destination_pos_local '
                             '(world XY). Default: 5m beyond ball in kick direction.')
    parser.add_argument("--robot_initial_xy", type=str, default="0.0,0.0",
                        help='World position of the robot at startup. The pelvis '
                             'position is hardcoded to this since the G1 IMU does '
                             'not report absolute world position. Set this if you '
                             'place the robot somewhere other than the origin.')
    parser.add_argument("--warmup_seconds", type=float, default=3.0,
                        help="Time to smoothly move from current pose to motion "
                             "frame 0 pose using onboard position control.")
    parser.add_argument("--kp_scale", type=float, default=1.0,
                        help="Scale factor on the training-time stiffness (kp). "
                             "Set to 0.7-0.8 for the first hardware run if you "
                             "want softer motors. >1.0 makes the robot stiffer.")
    parser.add_argument("--kd_scale", type=float, default=1.0,
                        help="Scale factor on the training-time damping (kd). "
                             "Same advice as --kp_scale.")
    parser.add_argument("--abort_tilt_deg", type=float, default=60.0,
                        help="Auto-disable motors if the pelvis tilts more than "
                             "this many degrees off vertical. Set to 90+ to "
                             "disable, lower than 60 for tighter safety.")
    parser.add_argument("--joint_clip_margin", type=float, default=0.05,
                        help="Safety margin (radians) inside each joint's "
                             "position limit. Target positions are clipped to "
                             "[lim_min + margin, lim_max - margin].")
    parser.add_argument("--damping_kd", type=float, default=5.0,
                        help="Damping coefficient used during soft stops "
                             "(safety abort or Ctrl+C). Higher = more rigid "
                             "collapse, lower = floppier. 5.0 is a sensible "
                             "default for the G1.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Read sensors and run policy, but do NOT send motor "
                             "commands. Use this first to verify everything.")
    parser.add_argument("--damping_only", action="store_true",
                        help="Enable motors in DAMPING mode only (kp=0, "
                             "kd=--damping_kd), do not warm up, do not run the "
                             "policy. Hold for --damping_seconds. Use this as "
                             "Stage 0.5 to verify your kd value is safe BEFORE "
                             "letting the policy push positions.")
    parser.add_argument("--damping_seconds", type=float, default=5.0,
                        help="Duration of --damping_only mode.")
    parser.add_argument("--execute", action="store_true",
                        help="Required to actually run the policy. Without this, "
                             "the script will not run the full kick motion.")
    parser.add_argument("--suspended", action="store_true",
                        help="Tell the script the robot is suspended (no floor "
                             "contact). Skips the abort-on-fall safety check.")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum control steps (500 = 10s at 50Hz).")
    args = parser.parse_args()

    if not (args.dry_run or args.damping_only or args.execute):
        sys.exit(
            "ERROR: must pass one of:\n"
            "  --dry_run      (safest — no motor commands)\n"
            "  --damping_only (motors enabled, damping only, no policy)\n"
            "  --execute      (full deployment — runs the policy)\n"
            "For your first run, use --dry_run."
        )

    ball_xy = tuple(float(v) for v in args.ball_xy.split(","))
    robot_xy = tuple(float(v) for v in args.robot_initial_xy.split(","))

    BALL_RADIUS = 0.11
    TARGET_POINT_WORLD = np.array(
        [ball_xy[0], ball_xy[1], BALL_RADIUS], dtype=np.float32
    )

    # Default destination: 5m beyond the ball along the (robot → ball) vector.
    if args.destination_xy is not None:
        dest_xy = tuple(float(v) for v in args.destination_xy.split(","))
        TARGET_DEST_WORLD = np.array(
            [dest_xy[0], dest_xy[1], BALL_RADIUS], dtype=np.float32
        )
    else:
        kick_vec = np.array(ball_xy) - np.array(robot_xy)
        kick_dir = kick_vec / (np.linalg.norm(kick_vec) + 1e-9)
        dest_xy = np.array(ball_xy) + 5.0 * kick_dir
        TARGET_DEST_WORLD = np.array(
            [dest_xy[0], dest_xy[1], BALL_RADIUS], dtype=np.float32
        )
    print(f"[deploy] Ball  world XY: {ball_xy}")
    print(f"[deploy] Robot world XY: {robot_xy}")
    print(f"[deploy] Destination XY: ({TARGET_DEST_WORLD[0]:.3f}, {TARGET_DEST_WORLD[1]:.3f})")

    # ── Load policy + precompute motion ──────────────────────────────────
    onnx_path = resolve_policy_path(
        local_path=args.onnx,
        hf_repo=args.hf_repo,
        hf_file=args.hf_file,
        cache_dir=args.hf_cache_dir,
    )
    assets = load_policy(onnx_path)
    motion = precompute_motion_table(assets)

    # ── Initialize Unitree DDS channel ───────────────────────────────────
    print(f"[deploy] Initializing Unitree SDK on interface '{args.network_interface}'")
    ChannelFactoryInitialize(0, args.network_interface)

    observer = G1StateObserver()
    print("[deploy] Waiting for first LowState message...")
    observer.wait_for_first_message(timeout_s=10.0)
    print("[deploy] LowState received. Sensors online.")

    # ── Sanity check sensor readings before any motor command ─────────
    initial_quat = observer.base_quat_wxyz()
    initial_q = observer.joint_pos_sdk_order()
    if not np.all(np.isfinite(initial_quat)) or np.allclose(initial_quat, 0):
        sys.exit(f"ERROR: invalid IMU quaternion {initial_quat}. Robot IMU "
                 "not initialized or SDK convention mismatched. Aborting.")
    if not np.all(np.isfinite(initial_q)):
        sys.exit(f"ERROR: motor encoders report NaN/inf in initial_q. Aborting.")
    # Verify robot is roughly upright before we enable anything
    g0 = projected_gravity_body(initial_quat)
    if g0[2] > -0.7:
        tilt = np.rad2deg(np.arccos(-g0[2]))
        sys.exit(f"ERROR: robot is tilted {tilt:.1f}° at startup "
                 f"(gravity_z_body={g0[2]:.3f}). Stand it upright first.")
    print(f"[deploy] Initial pose check OK (tilt {np.rad2deg(np.arccos(-g0[2])):.1f}°)")

    publisher = G1MotorPublisher() if not args.dry_run else None
    if args.dry_run:
        print("[deploy] DRY RUN — no motor commands will be sent.")
    else:
        print("[deploy] EXECUTE mode — motors will be commanded. Have your E-stop ready.")
        input("       Press Enter to start the warm-up sequence... ")

    # ── Warm-up: smoothly drive joints from current pose to motion frame 0 ─
    motion_init_q_policy = motion["joint_pos"][0].copy()
    motion_init_q_sdk = motion_init_q_policy[assets.sdk_to_policy]
    current_q_sdk = observer.joint_pos_sdk_order().copy()

    if not args.dry_run:
        kp_sdk = assets.joint_stiffness[assets.sdk_to_policy].copy()
        kd_sdk = assets.joint_damping[assets.sdk_to_policy].copy()
        n_warmup_steps = int(args.warmup_seconds * CONTROL_HZ)
        print(f"[deploy] Warming up for {args.warmup_seconds:.1f}s ({n_warmup_steps} steps)")
        for k in range(n_warmup_steps):
            alpha = (k + 1) / n_warmup_steps
            target_q_sdk = (1.0 - alpha) * current_q_sdk + alpha * motion_init_q_sdk
            publisher.send_position(target_q_sdk, kp_sdk, kd_sdk)
            time.sleep(CONTROL_DT)
        print("[deploy] Warm-up complete — robot at motion frame 0 pose.")
        input("       Press Enter to start the policy execution... ")

    # ── LSTM state and observation history ────────────────────────────────
    h = np.zeros((2, 1, 128), dtype=np.float32)
    c = np.zeros((2, 1, 128), dtype=np.float32)
    time_step = np.zeros((1, 1), dtype=np.float32)
    last_action = np.zeros(NUM_MOTORS, dtype=np.float32)

    DEFAULT_Q = assets.default_joint_pos          # policy order
    ACTION_SCALE = assets.action_scale            # policy order
    KP = assets.joint_stiffness * args.kp_scale   # policy order, scaled
    KD = assets.joint_damping   * args.kd_scale   # policy order, scaled
    TORQUE_LIMITS = assets.torque_limits_policy   # policy order

    # Joint position limits in POLICY order (re-index from SDK order).
    J_MIN_POLICY = G1_JOINT_MIN_SDK[assets.policy_to_sdk] + args.joint_clip_margin
    J_MAX_POLICY = G1_JOINT_MAX_SDK[assets.policy_to_sdk] - args.joint_clip_margin

    # Convert abort tilt to gravity_z threshold:
    # gravity_z = -cos(tilt_angle). Tilt of 0° → grav_z=-1; tilt of 60° → grav_z=-0.5.
    ABORT_GRAV_Z = -float(np.cos(np.deg2rad(args.abort_tilt_deg)))

    print(f"[deploy] kp scale: {args.kp_scale} (max kp={KP.max():.1f})")
    print(f"[deploy] kd scale: {args.kd_scale} (max kd={KD.max():.3f})")
    print(f"[deploy] Abort threshold: tilt > {args.abort_tilt_deg}° "
          f"(gravity_z > {ABORT_GRAV_Z:.3f})")
    print(f"[deploy] Joint position margin: ±{args.joint_clip_margin:.3f} rad inside limits")

    # ── Main control loop ────────────────────────────────────────────────
    print("[deploy] Entering control loop...")
    try:
        for step in range(args.max_steps):
            loop_start = time.time()

            # Look up motion data at frame T (matches sim2sim behavior).
            t_now = int(min(step, motion["total"] - 1))
            cmd_joint_pos = motion["joint_pos"][t_now]
            cmd_joint_vel = motion["joint_vel"][t_now]
            cmd_anchor_av = motion["anchor_av"][t_now]

            # ── Read sensors ────────────────────────────────────────────
            base_quat = observer.base_quat_wxyz()
            base_angvel_body = observer.base_ang_vel_body()
            joint_q_sdk = observer.joint_pos_sdk_order()
            joint_dq_sdk = observer.joint_vel_sdk_order()
            joint_q_policy = joint_q_sdk[assets.policy_to_sdk]
            joint_dq_policy = joint_dq_sdk[assets.policy_to_sdk]

            # ── Hardcoded robot world position (G1 IMU does not give it) ─
            # All world-frame transforms use this fixed position. As long as
            # the robot stays roughly on its mark this is accurate enough.
            pelvis_world = np.array([robot_xy[0], robot_xy[1], 0.79], dtype=np.float32)

            # ── Build the 160-dim observation ──────────────────────────
            command = np.concatenate([cmd_joint_pos, cmd_joint_vel])    # 58
            proj_grav = projected_gravity_body(base_quat)               # 3
            motion_ref_av = cmd_anchor_av                               # 3
            base_av = base_angvel_body                                  # 3
            jp_rel = joint_q_policy - DEFAULT_Q                         # 29
            jv = joint_dq_policy                                         # 29
            last_act = last_action                                       # 29
            target_ball = world_point_in_pelvis_frame(
                TARGET_POINT_WORLD, pelvis_world, base_quat
            )                                                            # 3
            target_dest = world_point_in_pelvis_frame(
                TARGET_DEST_WORLD, pelvis_world, base_quat
            )                                                            # 3

            obs = np.concatenate([
                command, proj_grav, motion_ref_av, base_av,
                jp_rel, jv, last_act, target_ball, target_dest,
            ]).astype(np.float32)[None, :]

            # ── Inference ──────────────────────────────────────────────
            outputs = assets.sess.run(
                None,
                {"obs": obs, "h_in": h, "c_in": c, "time_step": time_step},
            )
            actions = outputs[0][0]
            h = outputs[1]
            c = outputs[2]
            ts_total = outputs[-1][0, 0]
            time_step = np.clip(time_step + 1, 0, ts_total - 1)

            # ── Action clipping and target joint position ──────────────
            raw_actions = np.clip(actions, -10.0, 10.0)
            target_q_policy = DEFAULT_Q + raw_actions * ACTION_SCALE
            # Safety clip target to within each joint's physical limits.
            # The motor firmware also enforces these, but pre-clipping avoids
            # the PD producing huge torque pushing against a hard stop.
            target_q_policy = np.clip(target_q_policy, J_MIN_POLICY, J_MAX_POLICY)
            last_action = actions.copy()

            # ── PD torque, clipped to per-joint effort limits ──────────
            tau_policy = KP * (target_q_policy - joint_q_policy) - KD * joint_dq_policy
            tau_policy = np.clip(tau_policy, -TORQUE_LIMITS, TORQUE_LIMITS)
            tau_sdk = tau_policy[assets.sdk_to_policy]

            # ── Safety: abort if base orientation indicates a fall ─────
            if not args.suspended:
                gravity_z_body = proj_grav[2]
                if gravity_z_body > ABORT_GRAV_Z:
                    tilt_deg = np.rad2deg(np.arccos(-gravity_z_body))
                    print(f"[deploy] ABORT at step {step}: pelvis tilted "
                          f"{tilt_deg:.1f}° (limit {args.abort_tilt_deg:.0f}°). "
                          f"Switching to damping mode (soft fall).")
                    if publisher is not None:
                        # Soft stop: keep motors enabled with damping only so
                        # the robot collapses gently, not as a rag doll.
                        for _ in range(int(0.5 * CONTROL_HZ)):  # 500ms of damping
                            publisher.damping_all(kd_damp=args.damping_kd)
                            time.sleep(CONTROL_DT)
                        publisher.disable_all()
                    return

            # ── Send torque (or just print, in dry run) ────────────────
            if publisher is not None:
                publisher.send_torque(tau_sdk)

            if step % 25 == 0:
                print(
                    f"  step={step:4d}  t={step * CONTROL_DT:.2f}s  "
                    f"frame={t_now:4d}  act_std={np.std(actions):.3f}  "
                    f"|tau|max={np.abs(tau_policy).max():.1f}Nm  "
                    f"grav_z={proj_grav[2]:+.3f}"
                )

            # ── Maintain 50 Hz ─────────────────────────────────────────
            elapsed = time.time() - loop_start
            if elapsed < CONTROL_DT:
                time.sleep(CONTROL_DT - elapsed)
            elif elapsed > CONTROL_DT * 1.5:
                print(f"[deploy] WARNING: loop step {step} took {elapsed*1000:.1f}ms "
                      f"(target {CONTROL_DT*1000:.1f}ms)")

    except KeyboardInterrupt:
        print("\n[deploy] Interrupted by user — switching to damping mode for soft fall.")
        if publisher is not None:
            # 500 ms of damping so the robot collapses gently rather than
            # going limp instantly when Ctrl+C hits mid-motion.
            try:
                for _ in range(int(0.5 * CONTROL_HZ)):
                    publisher.damping_all(kd_damp=args.damping_kd)
                    time.sleep(CONTROL_DT)
            except Exception:
                pass
    finally:
        if publisher is not None:
            publisher.disable_all()
            print("[deploy] Motors fully disabled. Done.")


if __name__ == "__main__":
    main()
