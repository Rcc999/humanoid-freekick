"""
Sim2Sim: Run the exported CR7 free kick ONNX policy in Mujoco.

Usage:
    python scripts/sim2sim_mujoco.py \
        --onnx /tmp/policy_19000_cr7.onnx \
        --mjcf  third_party/HumanoidSoccer/source/whole_body_tracking/soccer/assets/unitree_description/mjcf/g1.xml

Requirements:
    pip install mujoco onnxruntime onnx numpy
"""

import argparse
import time
import numpy as np
import onnx
import onnxruntime as ort
import mujoco
import mujoco.viewer

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--onnx",  required=True, help="Path to exported ONNX policy")
parser.add_argument("--mjcf",  required=True, help="Path to g1.xml Mujoco model")
parser.add_argument("--steps", type=int, default=500, help="Max sim steps (50 Hz, so 500 = 10s)")
parser.add_argument("--gain_scale", type=float, default=1.0, help="Scale PD gains (try 0.5 if robot falls)")
parser.add_argument("--freeze_motion", action="store_true", help="Freeze motion reference at frame 0 (balance test)")
parser.add_argument("--headless", action="store_true", help="Run without viewer")
parser.add_argument("--ball_xy", type=str, default=None,
                    help='Override ball XY position as "x,y" (world frame, meters). '
                         'Default: motion last-anchor XY. Try "2.0,0.0" to place '
                         'ball 2m in front of robot.')
parser.add_argument("--record", type=str, default=None,
                    help="Save sim as mp4 to this path (e.g. /tmp/sim.mp4). "
                         "Renders via offscreen renderer; works with or without viewer.")
parser.add_argument("--record_fps", type=int, default=50,
                    help="FPS for recorded video (matches control rate).")
parser.add_argument("--noise", action="store_true",
                    help="Inject training-time sensor noise into observations. "
                         "Use this to validate hardware-readiness — if the policy "
                         "still works with this on, it'll likely transfer to real robot.")
parser.add_argument("--no_ball", action="store_true",
                    help="Skip injecting a physical ball into MuJoCo. The policy "
                         "still receives target_point_pos in the obs (using "
                         "--ball_xy or the motion's natural endpoint) but the "
                         "foot swings through empty air. Mirrors the hardware "
                         "'no ball' bring-up stage.")
args = parser.parse_args()
_ball_override = None
if args.ball_xy is not None:
    _ball_override = tuple(float(v) for v in args.ball_xy.split(","))
    assert len(_ball_override) == 2, "--ball_xy must be 'x,y'"

# ─── LOAD ONNX METADATA ───────────────────────────────────────────────────────
print("[sim2sim] Loading ONNX...")
onnx_model = onnx.load(args.onnx)
meta = {p.key: p.value for p in onnx_model.metadata_props}

POLICY_JOINT_NAMES  = meta["joint_names"].split(",")       # 29 joints in policy order
JOINT_STIFFNESS     = np.array(meta["joint_stiffness"].split(","), dtype=np.float32)
JOINT_DAMPING       = np.array(meta["joint_damping"].split(","),   dtype=np.float32)
# Applied after args are parsed — scaled below
DEFAULT_JOINT_POS   = np.array(meta["default_joint_pos"].split(","), dtype=np.float32)
ACTION_SCALE        = np.array(meta["action_scale"].split(","),    dtype=np.float32)
ANCHOR_BODY_NAME    = meta["anchor_body_name"]              # "torso_link"
BODY_NAMES          = meta["body_names"].split(",")         # 14 tracked bodies

print(f"  Policy joints ({len(POLICY_JOINT_NAMES)}): {POLICY_JOINT_NAMES[:5]}...")
print(f"  Anchor body: {ANCHOR_BODY_NAME}")
print(f"  Tracked bodies: {BODY_NAMES}")

# ─── LOAD MUJOCO MODEL (inject soccer ball) ───────────────────────────────────
print("[sim2sim] Loading Mujoco G1 model...")

# Inject a soccer ball into the XML as a free body.
# The XY is a placeholder — overridden below from the ONNX motion data.
# In training the ball is placed at the motion's LAST frame anchor (torso)
# position — that's where the robot lands at the end of the CR7 kick.
BALL_RADIUS = 0.11
BALL_START_XY = (0.29, 0.13)   # placeholder; replaced after warm-start

import os as _os
mjcf_dir = _os.path.dirname(_os.path.abspath(args.mjcf))

with open(args.mjcf) as f:
    xml_str = f.read()

# Resolve relative meshdir to absolute so from_xml_string can find the meshes
import re as _re
def _abs_meshdir(m):
    rel = m.group(1)
    abs_path = _os.path.normpath(_os.path.join(mjcf_dir, rel))
    return f'meshdir="{abs_path}"'
xml_str = _re.sub(r'meshdir="([^"]+)"', _abs_meshdir, xml_str)

# ── Strip MuJoCo-specific joint dynamics not present in IsaacSim training ──
# frictionloss: Coulomb joint friction (not in IsaacSim PhysX) — adds drag the
#   policy never saw and can't compensate via PD.
# NOTE: armature is intentionally KEPT — it stabilizes MuJoCo's explicit
#   integrator under high PD gains (kp=99). Removing it causes NaN/divergence.
# NOTE: We previously stripped frictionloss and softened foot-floor contacts
# trying to fix what we thought was a contact mechanics gap. The real bug was
# in base_ang_vel (double-rotation of body-frame qvel[3:6]). With that fixed,
# the contact workarounds are no longer needed and likely weaken the kick.
# Keeping the integrator=implicitfast as it matches IsaacSim's implicit PD.

# ── Inject a wide-view tracking camera so recordings show whole scene ───
# The default "track" camera in MuJoCo can be too close. We add a dedicated
# camera "wide_track" that follows the pelvis from a distance so both the
# robot and the kicked ball stay in frame.
_wide_cam = '<camera name="wide_track" mode="fixed" pos="4.5 -1.5 1.8" xyaxes="0 1 0 -0.4 0 0.92" fovy="55"/>'
if "<worldbody>" in xml_str and 'name="wide_track"' not in xml_str:
    xml_str = xml_str.replace("<worldbody>", f"<worldbody>\n    {_wide_cam}", 1)

# ── Ensure offscreen framebuffer is large enough for 1280x720 video ─────
# Default MuJoCo framebuffer is 640x480. The XML may already have a <global>
# element inside <visual>; in that case we add/replace the offwidth/offheight
# attributes on that existing element instead of inserting a duplicate.
def _patch_global(m):
    inner = m.group(1)  # attributes string inside <global ... />
    # Replace existing offwidth/offheight, or add them
    if 'offwidth' in inner:
        inner = _re.sub(r'offwidth="[^"]+"', 'offwidth="1280"', inner)
    else:
        inner = inner.rstrip() + ' offwidth="1280"'
    if 'offheight' in inner:
        inner = _re.sub(r'offheight="[^"]+"', 'offheight="720"', inner)
    else:
        inner = inner.rstrip() + ' offheight="720"'
    return f'<global {inner.strip()}/>'

if _re.search(r'<global\s+[^/>]*/>', xml_str):
    xml_str = _re.sub(r'<global\s+([^/>]*)/>', _patch_global, xml_str, count=1)
elif "<visual>" in xml_str:
    xml_str = xml_str.replace(
        "<visual>", '<visual>\n    <global offwidth="1280" offheight="720"/>', 1,
    )
else:
    xml_str = _re.sub(
        r"(<mujoco[^>]*>)",
        r'\1\n  <visual>\n    <global offwidth="1280" offheight="720"/>\n  </visual>',
        xml_str, count=1,
    )

# ── Switch MuJoCo to implicit integration ──
# IsaacSim's PhysX uses implicit PD integration → stable & damped under high
# kp. MuJoCo defaults to semi-implicit Euler which is less forgiving of the
# training-time gain values during single-leg balance transitions (sidestep).
# implicitfast is the recommended high-performance implicit option.
if "<option" in xml_str:
    xml_str = _re.sub(
        r'<option([^/>]*?)/?>',
        lambda m: f'<option{m.group(1)} integrator="implicitfast"/>'
        if 'integrator' not in m.group(1) else m.group(0),
        xml_str, count=1,
    )
else:
    xml_str = xml_str.replace(
        "<mujoco", '<mujoco', 1
    )
    xml_str = xml_str.replace(
        "</compiler>",
        '</compiler>\n  <option integrator="implicitfast"/>', 1,
    )
print(f"  Set integrator=implicitfast (matches IsaacSim implicit PD)")

if args.no_ball:
    print("  --no_ball: skipping ball injection. Policy will see the ball's "
          "target_point_pos as if it were at --ball_xy but no physical ball "
          "exists. Foot will swing through empty air.")
else:
    soccer_ball_xml = f"""
      <body name="soccer_ball" pos="{BALL_START_XY[0]} {BALL_START_XY[1]} {BALL_RADIUS}">
        <freejoint name="soccer_ball_joint"/>
        <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}"
              mass="0.43" friction="0.6 0.005 0.0001"
              condim="6" solimp="0.95 0.99 0.001" solref="0.02 1"/>
      </body>
    """
    # Insert into first </worldbody> (robot body), not the second one (floor/lights)
    xml_str = xml_str.replace("</worldbody>", soccer_ball_xml + "\n    </worldbody>", 1)

model = mujoco.MjModel.from_xml_string(xml_str)
data  = mujoco.MjData(model)

print(f"  Mujoco nq={model.nq} nv={model.nv} nu={model.nu}")

# ─── JOINT LAYOUT ─────────────────────────────────────────────────────────────
# qpos layout: [root_pos(3), root_quat(4), joints(N_joints), ball_pos(3), ball_quat(4)]
# qvel layout: [root_vel(3), root_angvel(3), joint_vels(N_joints), ball_vel(3), ball_angvel(3)]
# g1.xml: floating_base_joint (freejoint) + 29 revolute joints + soccer_ball (freejoint)
N_JOINTS = len(POLICY_JOINT_NAMES)   # 29
QPOS_JOINTS_START = 7                # after root 7-DOF freejoint
QPOS_JOINTS_END   = QPOS_JOINTS_START + N_JOINTS
QVEL_JOINTS_START = 6                # after root 6-DOF velocity
QVEL_JOINTS_END   = QVEL_JOINTS_START + N_JOINTS
QPOS_BALL_START   = QPOS_JOINTS_END  # ball xyz starts right after joints
QVEL_BALL_START   = QVEL_JOINTS_END

# ─── JOINT ORDER MAPPING ──────────────────────────────────────────────────────
# Map policy joint index → Mujoco joint index (within the 29-joint block).
mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                  for i in range(model.njnt)
                  if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]

print(f"  Mujoco actuated joints ({len(mj_joint_names)}): {mj_joint_names[:5]}...")

policy_to_mj = []
for pname in POLICY_JOINT_NAMES:
    if pname in mj_joint_names:
        policy_to_mj.append(mj_joint_names.index(pname))
    else:
        raise ValueError(f"Policy joint '{pname}' not found in Mujoco model. "
                         f"Available: {mj_joint_names}")

policy_to_mj = np.array(policy_to_mj, dtype=np.int32)
print(f"  Joint mapping: OK (first 5: {policy_to_mj[:5]})")

# ─── READ PER-JOINT TORQUE LIMITS FROM MJCF ────────────────────────────────
# Match IsaacSim's effort_limit_sim by reading actuatorfrcrange from the XML.
# Without these clips, the policy can command extreme torques (well beyond
# what the real robot can produce) which destabilizes single-leg balance.
# Reference: unitree_rl_gym & GMT sim2sim both clip torque to these limits.
TORQUE_LIMITS = np.zeros(len(POLICY_JOINT_NAMES), dtype=np.float32)
for pi, pname in enumerate(POLICY_JOINT_NAMES):
    mj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pname)
    # actuator_frcrange: model.jnt_range and similar. For joint range it's
    # `model.jnt_actfrcrange[mj_jnt_id]` returning (min, max).
    if hasattr(model, "jnt_actfrcrange"):
        lo, hi = model.jnt_actfrcrange[mj_jnt_id]
    else:
        # Fallback to symmetric range from XML actuatorfrcrange attribute.
        lo, hi = -1e6, 1e6
    # If the range is invalid (both zero), default to a high cap
    if hi <= 0 or hi == lo:
        hi = 1000.0
    TORQUE_LIMITS[pi] = float(hi)
print(f"  Torque limits per joint (first 5): {TORQUE_LIMITS[:5]}")
print(f"  Torque limits (min,max): ({TORQUE_LIMITS.min():.1f}, {TORQUE_LIMITS.max():.1f})")

# ─── ONNX RUNTIME SESSION ─────────────────────────────────────────────────────
print("[sim2sim] Creating ONNX session...")
sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def quat_inv(q):
    """Invert a quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_rotate(q, v):
    """Rotate vector v by quaternion q [w, x, y, z]."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    rotated = quat_mul(quat_mul(q, qv), quat_inv(q))
    return rotated[1:]

def get_mj_quat(data):
    """Root quaternion from Mujoco as [w, x, y, z]."""
    # Mujoco stores as [w, x, y, z] in qpos[3:7]
    return data.qpos[3:7].copy()

def projected_gravity(data):
    """Gravity vector [0,0,-1] projected into robot body frame."""
    q = get_mj_quat(data)
    g_world = np.array([0.0, 0.0, -1.0])
    return quat_rotate(quat_inv(q), g_world)

def base_ang_vel_body(data):
    """Root angular velocity in robot body frame.

    IMPORTANT: For MuJoCo free joints, qvel[3:6] is ALREADY in the local
    body frame (quaternion tangent space convention). Do NOT apply an
    inverse-quaternion rotation here — that would double-rotate and feed
    the policy garbage. This was the long-standing bug that made sim2sim
    fail with wild act_std and immediate falls.
    Ref: https://mujoco.readthedocs.io free-joint qvel convention.
    """
    return data.qvel[3:6].copy()

def get_joint_pos_mj(data):
    """All joint positions in Mujoco XML order (29 joints)."""
    return data.qpos[QPOS_JOINTS_START:QPOS_JOINTS_END].copy()

def get_joint_vel_mj(data):
    """All joint velocities in Mujoco XML order (29 joints)."""
    return data.qvel[QVEL_JOINTS_START:QVEL_JOINTS_END].copy()

def get_policy_joint_pos(data):
    """Joint positions in POLICY order."""
    qpos_mj = get_joint_pos_mj(data)
    return qpos_mj[policy_to_mj]

def get_policy_joint_vel(data):
    """Joint velocities in POLICY order."""
    qvel_mj = get_joint_vel_mj(data)
    return qvel_mj[policy_to_mj]

# ── World-fixed reference points (set once at episode start) ─────────────
# These match training semantics: in IsaacLab, target_point_pos and
# target_destination_pos_local both transform a FIXED WORLD POINT into the
# CURRENT pelvis frame at every step. As the robot moves/rotates, these
# values change — that's how the policy knows where the target is relative
# to itself.
TARGET_POINT_WORLD = np.array(
    [BALL_START_XY[0], BALL_START_XY[1], BALL_RADIUS], dtype=np.float32
)
# Destination: 5m forward of the robot's initial pose at ball height.
# Matches training where destination_pos is a fixed world point per episode.
TARGET_DEST_WORLD = np.array([5.0, 0.0, BALL_RADIUS], dtype=np.float32)


def world_point_in_pelvis_frame(data, point_world):
    """Transform a fixed world-space point into the current pelvis frame.

    This is the IsaacLab convention: (target_world - pelvis_world) rotated
    by inverse(pelvis_quat). Result changes as robot translates/rotates.
    """
    pelvis_world = data.qpos[0:3].copy()
    q_pelvis = get_mj_quat(data)
    delta = point_world - pelvis_world
    return quat_rotate(quat_inv(q_pelvis), delta)


def get_ball_pos_pelvis_frame(data):
    """Ball CURRENT position in robot pelvis local frame.

    When --no_ball is set, no physical ball exists in MuJoCo (QPOS_BALL_START
    would be invalid). Fall back to the fixed TARGET_POINT_WORLD so the
    policy still gets a consistent target_point_pos observation — mirrors
    the hardware deployment where there's no ball tracker.
    """
    if args.no_ball:
        return world_point_in_pelvis_frame(data, TARGET_POINT_WORLD)
    ball_world = data.qpos[QPOS_BALL_START:QPOS_BALL_START+3].copy()
    pelvis_world = data.qpos[0:3].copy()
    q_pelvis = get_mj_quat(data)
    delta = ball_world - pelvis_world
    return quat_rotate(quat_inv(q_pelvis), delta)

# ─── SET INITIAL ROBOT POSE ───────────────────────────────────────────────────
def reset_robot(data, init_joint_pos=None):
    """Place robot at motion frame-0 pose (or default if not yet known)."""
    mujoco.mj_resetData(model, data)
    qpos_init = data.qpos.copy()
    src = init_joint_pos if init_joint_pos is not None else DEFAULT_JOINT_POS
    for pi, mi in enumerate(policy_to_mj):
        qpos_init[QPOS_JOINTS_START + mi] = src[pi]
    # Place ball at correct offset (skipped when --no_ball is set since
    # there's no QPOS_BALL slot in that case).
    if not args.no_ball:
        qpos_init[QPOS_BALL_START:QPOS_BALL_START+3] = [BALL_START_XY[0], BALL_START_XY[1], BALL_RADIUS]
    data.qpos[:] = qpos_init
    mujoco.mj_forward(model, data)

# ─── SIMULATION LOOP ──────────────────────────────────────────────────────────
print("[sim2sim] Starting simulation...")

# ── Warm-start: get motion frame-0 joint positions from ONNX ──────────────
# Run a dummy forward pass with zero obs at t=0 to get the motion reference
# initial pose, then initialize the robot there instead of DEFAULT_JOINT_POS.
_dummy_obs = np.zeros((1, 160), dtype=np.float32)
_h0 = np.zeros((2, 1, 128), dtype=np.float32)
_c0 = np.zeros((2, 1, 128), dtype=np.float32)
_ts0 = np.zeros((1, 1), dtype=np.float32)
_out = sess.run(None, {"obs": _dummy_obs, "h_in": _h0, "c_in": _c0, "time_step": _ts0})
MOTION_INIT_JOINT_POS = _out[3][0].copy()   # joint_pos output at frame 0
MOTION_INIT_JOINT_VEL = _out[4][0].copy()   # joint_vel output at frame 0
print(f"  Motion frame-0 joint pos (first 5): {MOTION_INIT_JOINT_POS[:5]}")

_anchor_idx_warm = BODY_NAMES.index(ANCHOR_BODY_NAME) if ANCHOR_BODY_NAME in BODY_NAMES else 7
_ts_total = int(_out[-1][0, 0])   # time_step_total from ONNX output

# ── Pre-compute the entire motion data table ────────────────────────────
# Eliminates 1-step lag bug: at runtime step T, the obs needs motion data
# at frame T (NOT T-1 which is what saving previous ONNX output gave us).
# Pre-computing avoids the need for peek-ahead at runtime.
print(f"  Pre-computing motion table ({_ts_total} frames)...")
MOTION_JOINT_POS  = np.zeros((_ts_total, 29), dtype=np.float32)
MOTION_JOINT_VEL  = np.zeros((_ts_total, 29), dtype=np.float32)
MOTION_ANCHOR_AV  = np.zeros((_ts_total, 3),  dtype=np.float32)
MOTION_ANCHOR_POS = np.zeros((_ts_total, 3),  dtype=np.float32)
MOTION_ANCHOR_QUAT = np.zeros((_ts_total, 4), dtype=np.float32)
# Track right ankle (kicking foot for "_right" motions) over the whole motion
# so we can find where the foot actually STRIKES the ball, not where the torso
# ends up.
_R_ANKLE_IDX = BODY_NAMES.index("right_ankle_roll_link") if "right_ankle_roll_link" in BODY_NAMES else 6
_L_ANKLE_IDX = BODY_NAMES.index("left_ankle_roll_link") if "left_ankle_roll_link" in BODY_NAMES else 3
MOTION_R_ANKLE_POS = np.zeros((_ts_total, 3), dtype=np.float32)
MOTION_L_ANKLE_POS = np.zeros((_ts_total, 3), dtype=np.float32)
for t in range(_ts_total):
    _ts_q = np.array([[t]], dtype=np.float32)
    _o = sess.run(None, {"obs": _dummy_obs, "h_in": _h0, "c_in": _c0, "time_step": _ts_q})
    MOTION_JOINT_POS[t]   = _o[3][0]
    MOTION_JOINT_VEL[t]   = _o[4][0]
    MOTION_ANCHOR_POS[t]  = _o[5][0][_anchor_idx_warm, :]
    MOTION_ANCHOR_QUAT[t] = _o[6][0][_anchor_idx_warm, :]
    MOTION_ANCHOR_AV[t]   = _o[8][0][_anchor_idx_warm, :]
    MOTION_R_ANKLE_POS[t] = _o[5][0][_R_ANKLE_IDX, :]
    MOTION_L_ANKLE_POS[t] = _o[5][0][_L_ANKLE_IDX, :]
print(f"  Motion table built.")
print(f"  Motion frame-0 anchor pos: {MOTION_ANCHOR_POS[0]}")
print(f"  Motion frame-0 anchor quat (w,x,y,z): {MOTION_ANCHOR_QUAT[0]}")
print(f"  Motion last-frame anchor pos: {MOTION_ANCHOR_POS[-1]}")
print(f"  Motion last-frame anchor quat: {MOTION_ANCHOR_QUAT[-1]}")

# ── Identify the kick strike frame: frame where right ankle has max velocity ─
_r_ankle_vel = np.linalg.norm(np.diff(MOTION_R_ANKLE_POS, axis=0), axis=1)
_kick_frame = int(np.argmax(_r_ankle_vel))
_strike_xy = MOTION_R_ANKLE_POS[_kick_frame, :2]
print(f"  Right ankle max-velocity frame: {_kick_frame}/{_ts_total} "
      f"(t={_kick_frame * 0.02:.2f}s)")
print(f"  Right ankle XY at strike frame: ({_strike_xy[0]:.3f}, {_strike_xy[1]:.3f})")
print(f"  (Use --ball_xy '{_strike_xy[0]:.3f},{_strike_xy[1]:.3f}' to place ball at strike point)")

# Ball at motion's last-frame anchor XY (matches training behavior),
# unless overridden via --ball_xy.
if _ball_override is not None:
    BALL_START_XY = _ball_override
    _ball_xy = np.array(_ball_override, dtype=np.float32)
    print(f"  Ball position OVERRIDDEN by --ball_xy: {BALL_START_XY}")
else:
    _ball_xy = MOTION_ANCHOR_POS[-1, :2]
    BALL_START_XY = (float(_ball_xy[0]), float(_ball_xy[1]))
    print(f"  Ball placed at motion last-frame anchor XY: {BALL_START_XY}")

# Destination: 5m beyond ball in kick direction
_anchor_first = MOTION_ANCHOR_POS[0, :2]
_kick_dir = _ball_xy - _anchor_first
_kick_dir_norm = np.linalg.norm(_kick_dir)
if _kick_dir_norm > 1e-6:
    _kick_dir = _kick_dir / _kick_dir_norm
else:
    _kick_dir = np.array([1.0, 0.0])
_dest_xy = _ball_xy + 5.0 * _kick_dir
TARGET_DEST_WORLD = np.array([_dest_xy[0], _dest_xy[1], BALL_RADIUS], dtype=np.float32)
print(f"  Destination (5m beyond ball in kick dir): {TARGET_DEST_WORLD}")
TARGET_POINT_WORLD = np.array([BALL_START_XY[0], BALL_START_XY[1], BALL_RADIUS], dtype=np.float32)

reset_robot(data, init_joint_pos=MOTION_INIT_JOINT_POS)

# LSTM state
h = np.zeros((2, 1, 128), dtype=np.float32)
c = np.zeros((2, 1, 128), dtype=np.float32)
time_step = np.zeros((1, 1), dtype=np.float32)
last_actions = np.zeros(29, dtype=np.float32)

# Seed motion reference with actual frame-0 values so step-0 command is correct
joint_pos_ref    = MOTION_INIT_JOINT_POS.copy()
joint_vel_ref    = MOTION_INIT_JOINT_VEL.copy()
motion_ref_angvel_prev = np.zeros(3, dtype=np.float32)  # persists between steps

# Anchor body index in the 14-body ONNX output
try:
    anchor_idx = BODY_NAMES.index(ANCHOR_BODY_NAME)
except ValueError:
    anchor_idx = 7  # fallback: torso_link is typically index 7

SIM_DT   = model.opt.timestep   # Mujoco physics dt (typically 0.002)
CTRL_HZ  = 50.0                  # Policy runs at 50 Hz (decimation=4 × 0.005=0.02s)
CTRL_DT  = 1.0 / CTRL_HZ
STEPS_PER_CTRL = max(1, int(round(CTRL_DT / SIM_DT)))

JOINT_STIFFNESS = JOINT_STIFFNESS * args.gain_scale
JOINT_DAMPING   = JOINT_DAMPING   * args.gain_scale
print(f"  Physics dt={SIM_DT:.4f}s, Control dt={CTRL_DT:.4f}s, steps/ctrl={STEPS_PER_CTRL}")
print(f"  PD gain scale: {args.gain_scale}x  (kp max={JOINT_STIFFNESS.max():.1f}, kd max={JOINT_DAMPING.max():.3f})")

def run_step(step_idx):
    global h, c, time_step, last_actions, joint_pos_ref, joint_vel_ref, motion_ref_angvel_prev

    # ── Build 160-dim observation ──────────────────────────────────────────
    # Use the precomputed motion table indexed by CURRENT time_step (T),
    # NOT previous step's ONNX output (T-1). Fixes 1-step lag in command
    # and motion_ref_ang_vel that caused obs distribution mismatch.
    t_now = int(min(step_idx, _ts_total - 1))
    cur_joint_pos_ref = MOTION_JOINT_POS[t_now]
    cur_joint_vel_ref = MOTION_JOINT_VEL[t_now]
    cur_motion_ref_av = MOTION_ANCHOR_AV[t_now]

    # 1. command (58): [joint_pos_ref, joint_vel_ref] at frame T
    command = np.concatenate([cur_joint_pos_ref, cur_joint_vel_ref])    # (58,)

    # 2. projected_gravity (3)
    proj_grav = projected_gravity(data)                                  # (3,)

    # 3. motion_ref_ang_vel (3): anchor body ang vel at frame T (world frame)
    motion_ref_angvel = cur_motion_ref_av.copy()                        # (3,)

    # 4. base_ang_vel (3): robot angular velocity in body frame
    base_angvel = base_ang_vel_body(data)                               # (3,)

    # 5. joint_pos (29): relative to default
    jp_rel = get_policy_joint_pos(data) - DEFAULT_JOINT_POS             # (29,)

    # 6. joint_vel (29)
    jv = get_policy_joint_vel(data)                                      # (29,)

    # 7. last actions (29)
    acts = last_actions.copy()                                           # (29,)

    # 8. target_point_pos (3): ball's CURRENT world position transformed
    #    to current pelvis frame. Matches training where
    #    `target_point_pos = soccer_ball_pos.clone()` updated every step.
    target_ball = get_ball_pos_pelvis_frame(data)                       # (3,)

    # 9. target_destination_pos_local (3): FIXED world destination transformed
    #    to current pelvis frame. Matches training's target_destination_pos_local
    #    where destination is sampled once at episode reset and stays fixed.
    target_dest = world_point_in_pelvis_frame(data, TARGET_DEST_WORLD)   # (3,)

    # ── Optional sensor noise (matches training-time noise ranges) ──────
    # Training used UniformNoiseCfg on these obs terms. Adding the same noise
    # in deploy is the closest sim2sim approximation of real-hardware sensors.
    # If the policy still works with --noise enabled, hardware is much more
    # likely to succeed.
    if args.noise:
        proj_grav         = proj_grav         + np.random.uniform(-0.05, 0.05, size=3).astype(np.float32)
        motion_ref_angvel = motion_ref_angvel + np.random.uniform(-0.05, 0.05, size=3).astype(np.float32)
        base_angvel       = base_angvel       + np.random.uniform(-0.2,  0.2,  size=3).astype(np.float32)
        jp_rel            = jp_rel            + np.random.uniform(-0.01, 0.01, size=29).astype(np.float32)
        jv                = jv                + np.random.uniform(-0.5,  0.5,  size=29).astype(np.float32)

    obs = np.concatenate([
        command,        # 58
        proj_grav,      # 3
        motion_ref_angvel,  # 3
        base_angvel,    # 3
        jp_rel,         # 29
        jv,             # 29
        acts,           # 29
        target_ball,    # 3
        target_dest,    # 3
    ]).astype(np.float32)[None, :]  # (1, 160)

    # ── Run ONNX ───────────────────────────────────────────────────────────
    outputs = sess.run(None, {
        "obs":       obs,
        "h_in":      h,
        "c_in":      c,
        "time_step": time_step,
    })
    actions_out, h_out, c_out, jp_ref, jv_ref, bp_w, bq_w, blv_w, bav_w, ts_total = outputs

    # Update LSTM state and time step
    h = h_out
    c = c_out
    if not args.freeze_motion:
        time_step = np.clip(time_step + 1, 0, ts_total[0, 0] - 1)
    # freeze_motion keeps time_step=0 → same motion reference every step (balance test)

    # Update persistent motion reference for next step's obs
    joint_pos_ref          = jp_ref[0]                   # (29,)
    joint_vel_ref          = jv_ref[0]                   # (29,)
    motion_ref_angvel_prev = bav_w[0, anchor_idx, :].copy()  # (3,) — global, used next step

    # ── Compute target joint positions (PD applied inside physics loop) ──
    # Clip raw actions to [-10, 10] BEFORE scaling — matches GMT/unitree_rl_gym
    # reference impls. Prevents the policy from sending the robot to unreachable
    # joint targets when act_std spikes (the cause of our wild fall behavior).
    raw_actions = np.clip(actions_out[0], -10.0, 10.0)
    actions_scaled = raw_actions * ACTION_SCALE             # (29,) policy scale
    target_joint_pos = DEFAULT_JOINT_POS + actions_scaled   # (29,)

    last_actions = actions_out[0].copy()

    # ── Step physics (PD re-evaluated every physics step for stability) ───
    for _ in range(STEPS_PER_CTRL):
        # Recompute torques at physics rate (500 Hz) for numerical stability
        current_jp = get_policy_joint_pos(data)
        current_jv = get_policy_joint_vel(data)
        torques_policy_phys = (JOINT_STIFFNESS * (target_joint_pos - current_jp)
                               - JOINT_DAMPING  * current_jv)
        # Clip torques to per-joint effort limits — matches IsaacSim's
        # effort_limit_sim that was enforced during training. Without this,
        # the policy can command torques far beyond what the real robot can
        # produce, destabilizing the sim during dynamic motions like the kick.
        torques_policy_phys = np.clip(
            torques_policy_phys, -TORQUE_LIMITS, TORQUE_LIMITS
        )
        data.qfrc_applied[:] = 0.0
        for pi, mi in enumerate(policy_to_mj):
            data.qfrc_applied[QVEL_JOINTS_START + mi] = float(torques_policy_phys[pi])
        mujoco.mj_step(model, data)

    if step_idx % 50 == 0:
        pelvis_z = data.qpos[2]
        if args.no_ball:
            ball_str = "no-ball"
        else:
            ball_pos = data.qpos[QPOS_BALL_START:QPOS_BALL_START+3]
            ball_str = f"ball={ball_pos[:2]}"
        print(f"  step={step_idx:4d} t={data.time:.2f}s  pelvis_z={pelvis_z:.3f}m  "
              f"{ball_str}  act_std={np.std(actions_out[0]):.3f}")


# ── Optional video recording via offscreen renderer ──────────────────────
_video_writer = None
_renderer = None
if args.record is not None:
    try:
        import imageio
    except ImportError:
        raise SystemExit("--record requires `pip install imageio imageio-ffmpeg`")
    _renderer = mujoco.Renderer(model, height=720, width=1280)
    _video_writer = imageio.get_writer(args.record, fps=args.record_fps, codec="libx264", quality=8)
    print(f"[sim2sim] Recording to {args.record} at {args.record_fps} FPS")


def _capture_frame():
    if _video_writer is None:
        return
    # Prefer the injected wide-angle camera so the kick remains in frame.
    for cam in ("wide_track", "track"):
        try:
            _renderer.update_scene(data, camera=cam)
            break
        except Exception:
            continue
    else:
        _renderer.update_scene(data)
    _video_writer.append_data(_renderer.render())


if args.headless:
    for i in range(args.steps):
        run_step(i)
        _capture_frame()
    print("[sim2sim] Done.")
else:
    # launch_passive requires mjpython on macOS; try it, fall back to renderer
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            viewer.cam.distance  = 3.0
            viewer.cam.elevation = -20
            step = 0
            while viewer.is_running() and step < args.steps:
                t0 = time.time()
                run_step(step)
                _capture_frame()
                viewer.sync()
                elapsed = time.time() - t0
                if elapsed < CTRL_DT:
                    time.sleep(CTRL_DT - elapsed)
                step += 1
    except RuntimeError:
        print("[sim2sim] launch_passive failed — use: mjpython scripts/sim2sim_mujoco.py ...")
        print("[sim2sim] Falling back to offscreen-only mode")
        if _renderer is None:
            _renderer = mujoco.Renderer(model, height=720, width=1280)
        for i in range(args.steps):
            run_step(i)
            _capture_frame()
        print("[sim2sim] Done.")
    print("[sim2sim] Done.")

if _video_writer is not None:
    _video_writer.close()
    print(f"[sim2sim] Saved video to {args.record}")
