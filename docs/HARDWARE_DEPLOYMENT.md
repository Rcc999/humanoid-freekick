# Hardware deployment — CR7 free-kick policy on the Unitree G1

This document walks through deploying the trained policy on a real Unitree G1
(29-DOF version). It covers everything from prep to a successful kick.

> ⚠ **Read the SAFETY section before doing anything.** A humanoid robot
> falling can break itself or hurt someone. Treat this as a serious procedure.

---

## 0. What you have

- **Policy:** `policy_12000_football_stylized-001_right.onnx` — the 12k
  DR-fine-tuned policy that passed sim2sim + noise validation in MuJoCo.
  Hosted on Hugging Face at
  [`Shish999/cr7-freekick-g1`](https://huggingface.co/Shish999/cr7-freekick-g1)
  and auto-downloaded by the deployment script on first run.
- **Motion file:** `football_stylized-001_right.npz` (right-footed sidestep
  kick, embedded in the policy via the ONNX motion table — no separate
  file to copy).
- **Deployment script:** `scripts/deploy_g1_hardware.py`

The policy expects:
- 29 actuated joints in the G1's standard 29-DOF order
- 50 Hz control rate (20 ms per step)
- ~7.8 s of motion (390 frames at 50 Hz)
- Ball at a known fixed position relative to the robot

---

## 1. Safety setup (do FIRST — this is non-negotiable)

### Physical space
- **Floor:** flat, dry, non-slip
- **Clearance:** at least 5 m of empty space in the kick direction (see §3)
- **Padding:** crash mats around the robot if you have them
- **Bystanders:** nobody within 3 m of the robot during execution

### Mechanical safety
- **Safety harness:** rope + carabiners to the robot's lifting points,
  attached to a gantry or sturdy frame. For the first 3 hardware tests
  you must keep tension so the robot cannot fall.

### Emergency stops (three independent layers — use ALL of them)

| Layer | Mechanism | Latency | Survives software hang? |
|-------|-----------|---------|--------------------------|
| 1. **Hardware kill switch** | Physical button on the G1's back/torso. Cuts motor power directly. | < 50 ms | ✅ Yes |
| 2. **Joystick L2+B** | Unitree controller's software damping stop. Sends a damping command via the high-level path. | ~50 ms | ✅ Yes (independent process) |
| 3. **Ctrl+C in SSH** | Kills the Python deployment script; the `finally` block sends `motor_cmd[i].mode = 0` to disable every motor. | depends on script responsiveness | ❌ No |

**Always have a teammate holding the joystick during a run.** The Ctrl+C
in your SSH terminal is the *least* reliable E-stop — if the script
hangs or the network drops, it won't help.

In addition, the deployment script has a **built-in software abort**:
if the pelvis tilts more than `--abort_tilt_deg` (default 60°) off
vertical, it switches to **damping mode** for 500 ms before fully
disabling motors. Tunable for tighter safety:
`--abort_tilt_deg 45` aborts at a much smaller tilt.

### Hard disable vs damping — what the script does

There are two ways to stop a motor, and they have very different physical
consequences:

| Mode | Motor command | Robot behavior |
|------|---------------|----------------|
| **Hard disable** (`mode=0`) | No torque at all | Robot goes **fully limp** — collapses like a rag doll, joints can slam into stops |
| **Damping** (`mode=1, kp=0, kd>0`) | Resists motion proportional to joint velocity | Robot collapses **gently** — joints brake themselves as they move, gravity pulls the body down softly |

The Unitree controller's L2+B sends damping (not hard disable) — that's
why it's safer than yanking power. The deployment script does the same:

- **Safety abort (tilt detected):** 500 ms of damping → then hard disable
- **Ctrl+C while running:** 500 ms of damping → then hard disable
- **Clean end-of-run:** hard disable (motion is done, robot already stable)
- **Hardware kill button:** immediate hard cutoff — only use for serious emergencies

You can tune the damping strength with `--damping_kd` (default `5.0`). Higher = more rigid collapse, lower = floppier.

### Battery and power
- **Charge to 100%** before any test session
- Watch voltage during runs — Unitree G1 starts behaving erratically
  below ~24 V
- Have a charged backup battery ready

### Network
- Wire the G1 to your laptop via Ethernet. Note your interface name
  (e.g. `eth0`, `eno1`, `en0`) — you'll pass it to the script as
  `--network_interface`.
- Avoid Wi-Fi for control. Wi-Fi latency will break 50 Hz timing.

---

## 2. Software prerequisites

### On the robot's onboard computer (Jetson)

Connect via SSH and clone the repo:

```bash
ssh unitree@<g1-ip>
cd ~
git clone https://github.com/Rcc999/humanoid-freekick.git cr7
cd cr7
```

(If you've already cloned it, just `cd ~/cr7 && git pull` to fetch the
latest deployment files.)

> **If the repo is private** the Jetson will prompt for a GitHub username
> + Personal Access Token (PAT). Generate one at
> https://github.com/settings/tokens (scope: `repo`) and paste it as the
> password. Alternatively, clone via SSH after copying your SSH key to
> the Jetson:
> `git clone git@github.com:Rcc999/humanoid-freekick.git cr7`

Then set up a Python virtual environment with everything the deployment
needs:

```bash
# On the Jetson, from the repo root
bash scripts/deploy/setup_venv.sh
```

The script creates `~/cr7/.venv` and installs:
- `numpy`, `onnx`, `onnxruntime` — for policy inference
- `huggingface_hub` — to auto-download the policy from the HF repo
- `unitree_sdk2_python` — to talk to the robot's motors and IMU

Activate the venv whenever you want to run the deployment:

```bash
source ~/cr7/.venv/bin/activate
```

Verify the install:

```bash
python3 -c 'import onnxruntime, onnx, numpy, huggingface_hub, unitree_sdk2py; print("OK")'
```

You should see `OK`. If you see an ImportError, re-run `bash setup_venv.sh`
or install the missing package manually with `pip install <name>`.

### Robot mode
Put the G1 into **developer mode** so the built-in `sport_client` does NOT
move the motors. With the joystick:

1. `L1 + Up` to enter dev mode
2. The robot should sit slightly slack (no high-level controller running)
3. Confirm `rt/lowstate` is publishing and `rt/lowcmd` is open for writes

### Policy file

You do NOT need to copy the policy manually — `deploy_g1_hardware.py`
downloads it from Hugging Face on first run and caches it at
`~/.cache/huggingface/hub`. The default file is:

- **Repo:** [`Shish999/cr7-freekick-g1`](https://huggingface.co/Shish999/cr7-freekick-g1)
- **File:** `dr/2026-06-04_17-19-06/exported/policy_12000_football_stylized-001_right.onnx`

If the Jetson has no internet, download the policy on your laptop first
and `scp` only the ONNX file (the script + docs come via `git clone`):

```bash
# On your laptop (one-time download)
python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('Shish999/cr7-freekick-g1', \
                    'dr/2026-06-04_17-19-06/exported/policy_12000_football_stylized-001_right.onnx', \
                    local_dir='./')"

# Copy the ONNX into the cloned repo on the robot
scp dr/2026-06-04_17-19-06/exported/policy_12000_*.onnx unitree@<g1-ip>:~/cr7/

# On the Jetson, pass it via --onnx:
#   python3 ~/cr7/scripts/deploy_g1_hardware.py --onnx ~/cr7/policy_12000_*.onnx ...
```

---

## 3. Robot orientation and ball placement

### How the motion works
The motion file `football_stylized-001_right.npz` captures CR7's right-footed
sidestep kick. The robot:
1. Starts at world `(0, 0)` facing the world `+X` axis
2. **Turns ~90° to the right** during the run-up
3. **Kicks the ball in the world `-Y` direction**
4. Ends with the torso at world `(0.06, -1.45)`

In other words: **the kick goes to the robot's right.** This is NOT a
forward kick. If you mentally model "robot facing forward kicks straight
ahead", you'll set everything up wrong.

### Coordinate convention used by the script
- World origin = where you place the robot
- Robot starts facing world `+X`
- Ball is at world `(0.06, -1.45)` — 6 cm forward of the robot, 1.45 m
  to the robot's right
- Kick destination is automatically computed as 5 m beyond the ball in
  the kick direction (i.e. further down `-Y`)

### Setting up the test space

```
                                        clearance for kick (5+ m)
                                        ↓
                                        ↓
                                        ↓
                            BALL ●------------------> (kick direction)
                              ↑   (1.45 m to robot's right)
                              ↑
                              ↑
  ROBOT ▲ ──────  (facing +X "forward", ignore that the kick goes sideways)
   start

Top-down view. +X is "robot's forward", +Y is "robot's left".
```

**Steps to set up:**
1. Mark a point on the floor — call this the **robot origin** (e.g. tape an X)
2. Stand the robot at the origin, oriented so its "front" (chest direction)
   points along your chosen `+X` direction
3. Measure **6 cm forward, 1.45 m to the right** from the origin and place
   the ball there
4. Make sure there's at least 5 m of clear space continuing past the ball
   in the same direction (the kick will send the ball that way)

### If you really want a "forward" kick
You can rotate the world coordinate frame. The script doesn't care which
direction is which — the ball position is whatever you tell it. To make
the kick "go forward" from your viewpoint, place the robot so that the
direction you call "forward" lines up with the kick trajectory, and pass
the corresponding ball position. Example: if you stand the robot facing
"north" and want the kick to go "north", you need the ball "east" of the
robot (because the motion kicks to the right) — and the kick will go
"south" from the robot's body frame but "north" from your viewpoint
(provided the robot's initial yaw is rotated 180°). It's confusing —
easier to just use the natural setup above for your first deployment.

---

## 4. Bring-up sequence (DO IN ORDER)

### Stage 0 — Verify telemetry on the bench
Before any motor action, just read sensors.

```bash
# On the Jetson (motors will not be commanded)
# Policy auto-downloads from Hugging Face on first run.
python3 ~/cr7/scripts/deploy_g1_hardware.py \
    --network_interface eth0 \
    --dry_run
```

**Pass criteria:**
- "LowState received. Sensors online." appears
- Loop runs at 50 Hz with `|tau|max` printed reasonably (not NaN, not huge)
- `act_std` is roughly 0.3–1.5
- `grav_z` is around `-0.95` to `-1.0` (robot upright)

If any of these fails, **do not proceed**. Most likely cause: wrong
`--network_interface`, robot not in dev mode, or `unitree_sdk2_python`
not installed correctly.

### Stage 1 — Suspended robot, no ball
Hang the robot from the gantry so its feet are off the floor by 5–10 cm.
Make sure the harness has tension before enabling motors.

```bash
python3 ~/cr7/scripts/deploy_g1_hardware.py \
    --network_interface eth0 \
    --suspended \
    --execute
```

The script will:
1. Read sensors and confirm online
2. Print `EXECUTE mode` and wait for you to press Enter
3. **Warm up** for 3 s, smoothly moving joints from current pose to
   motion frame-0 pose (uses onboard PD, gentle motion)
4. Wait for another Enter press
5. **Run the policy** for up to 10 s

**Pass criteria:**
- Warm-up completes without joints fighting (robot looks calm)
- During the run, the legs swing through the kick motion
- `act_std` peaks around 1–2 and recovers, doesn't blow up to 10+
- No motor errors printed by the SDK

If the policy is destabilizing the robot in mid-air (it shouldn't, since
there's no ground), abort with Ctrl+C and inspect the joint mapping in
the script.

### Stage 2 — On harness, on ground, no ball
Lower the robot so its feet are on the floor but the harness still has
~2–5 cm of slack (will catch a fall but not lift the robot).

```bash
python3 ~/cr7/scripts/deploy_g1_hardware.py \
    --network_interface eth0 \
    --execute
```

(Note: no `--suspended` this time — the safety abort if the robot tips over
is now armed.)

**Pass criteria:**
- Robot stands during warm-up
- Robot executes the full kick motion (run-up, sidestep, kick swing,
  follow-through) without falling
- Safety abort doesn't trigger

Repeat this stage 3+ times. If it works every time, you're ready for the
ball.

### Stage 2.5 — On harness, on ground, NO physical ball, policy thinks ball is there

This is the cleanest "kick motion only" test. The policy doesn't perceive the
ball through any sensor — it's just told where the ball is via `--ball_xy`.
So if you pass a ball position but don't physically place one, the robot
executes the full kick motion into empty air. This lets you see the form
without any physical contact risk.

```bash
python3 ~/cr7/scripts/deploy_g1_hardware.py \
    --network_interface eth0 \
    --ball_xy 0.06,-1.45 \
    --execute
```

**Pass criteria:**
- Robot does run-up + sidestep + kick swing + follow-through, all without
  a ball present
- Right foot swings through the spot where the ball "should" be
- No fall, no abort

This is a great Stage to repeat several times before introducing the
physical ball — same control loop, no contact dynamics, free to refine.

### Stage 3 — On harness, with ball
Place the ball as described in §3. Keep the safety harness slack.

```bash
python3 ~/cr7/scripts/deploy_g1_hardware.py \
    --network_interface eth0 \
    --ball_xy 0.06,-1.45 \
    --execute
```

**Pass criteria:**
- Robot does the full motion
- Right foot strikes the ball at ~t=5 s (the motion's natural kick frame)
- Ball travels in the -Y direction (relative to robot's initial pose)
- Robot stays upright through follow-through

### Stage 4 — Free, with ball
After Stage 3 has succeeded several times, remove the harness. Same
command as above.

---

## 5. Tuning knobs

### `--ball_xy "X,Y"`
Where the script tells the policy the ball is. Try slightly different
values if the kick form looks wrong:

| Value | Effect |
|-------|--------|
| `0.06,-1.45` | Default (motion's natural endpoint) |
| `0.06,-1.32` | Closer to robot — kick happens earlier |
| `0.06,-1.70` | Further out — kick happens later |
| `0.05,-2.0` | Used in sim2sim testing — works but a bit late |

For hardware always start with the default and adjust by ≤0.2 m.

### `--robot_initial_xy "X,Y"`
The world position of the robot at startup. Defaults to `0,0`. You only
need to change this if you place the robot at a known offset and want
the ball coordinate in that same world frame.

### `--destination_xy "X,Y"`
Where to "aim" the kick. Defaults to 5 m beyond the ball in the kick
direction. The policy uses this for the `target_destination_pos_local`
observation. You can usually leave it.

### `--warmup_seconds 3.0`
Duration of the smooth move from current pose to motion frame-0 pose.
Increase if the warm-up looks jerky (max 5 s); decrease if the robot's
already in the right pose.

### `--max_steps 500`
Total number of policy steps (500 = 10 s at 50 Hz). The motion is ~7.8 s
so anything ≥ 400 is fine.

### `--kp_scale 1.0` and `--kd_scale 1.0`
Multiplicative scale on the training-time stiffness (kp) and damping
(kd) gains. By default we use the exact values from the policy ONNX
metadata (matching training). For the **first hardware run** consider
softer gains:

```bash
python3 ~/cr7/scripts/deploy_g1_hardware.py \
    --network_interface eth0 \
    --kp_scale 0.7 --kd_scale 0.8 \
    --suspended --execute
```

Softer gains make the policy less aggressive but also less able to
balance — only use scales <1.0 while suspended for the first test. Once
on the ground, return to 1.0.

### `--abort_tilt_deg 60.0`
Auto-abort threshold for the built-in fall safety. If the pelvis tilts
more than this many degrees off vertical, the script disables all
motors. Defaults to 60°. Set lower (e.g. 45°) for tighter safety during
early runs; set to 90 to effectively disable.

### `--joint_clip_margin 0.05`
Safety margin (radians) inside each joint's physical limit. Target
positions are clipped to `[limit_min + margin, limit_max - margin]`.
Default 0.05 rad (~3°). Increase to 0.1 for more buffer if the robot
hits joint stops during the motion.

---

## 6. Troubleshooting

### "TimeoutError: No LowState received within 5s"
- Wrong `--network_interface` — run `ip a` on the Jetson to list interfaces
- Robot not powered on or not in dev mode
- Firewall blocking DDS multicast

### Robot falls during warm-up
- Initial pose mismatch — the script tries to move joints to motion frame 0
  from wherever they are, but if they're very far (e.g. arms hanging down
  vs raised) the warm-up may struggle
- Solution: pre-position the robot manually so it's roughly in the
  motion's frame-0 pose before running the script

### Robot tilts and the script aborts
- The `gravity_z_body > -0.5` check fires when the robot is more than ~60°
  off vertical
- If this happens repeatedly during the kick: the kick-swing single-leg
  balance is too aggressive for hardware. Consider re-training with stronger
  push perturbation DR, or scale down the PD gains
- Add `--suspended` to disable the check while diagnosing

### `act_std` blows up to 5+
- Strong sign that an observation is off-distribution
- Most common cause: joint ordering mismatch between SDK and policy
  expectations. Verify `G1_SDK_JOINT_ORDER` in the script matches your
  firmware exactly
- Second most common: IMU quaternion convention. The script assumes
  `[w, x, y, z]` ordering as exposed by `unitree_sdk2py`. If your SDK
  version reports `[x, y, z, w]`, edit `G1StateObserver.base_quat_wxyz`

### Robot strikes the ball but it barely moves
- Same issue we saw in sim2sim: the kick is weaker on real contact
  mechanics than the policy was trained on
- Options: physically use a lighter ball for testing, accept the weaker
  kick, or do a quick fine-tune with higher contact friction in IsaacLab

### Loop takes >20 ms per step
- ONNX inference too slow on Jetson — try the GPU provider:
  `ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])`
- Or reduce LSTM hidden size in a re-trained policy

---

## 7. What to do if it works on the first try

1. Don't immediately do a second run — debrief first
2. Video the run if you didn't already
3. Save the telemetry (consider modifying the script to log all obs and
   actions to a CSV)
4. Note the exact ball placement that worked
5. Then iterate

---

## 8. What to do if it doesn't work

The two most likely failure modes on hardware (in order of likelihood):

1. **Joint ordering or quaternion convention mismatch.** The script
   assumes the SDK joint order in `G1_SDK_JOINT_ORDER`. If your firmware
   exposes them differently, the policy will command the wrong motors.
   Verify by manually moving one joint at a time and checking which
   `motor_state[i].q` changes.

2. **The 12k DR policy isn't robust enough for real motor latency.** The
   training DR included motor delay 0–40 ms. If your hardware has higher
   latency (or jitter > 40 ms), the policy may oscillate. Solutions:
   - Measure actual latency with `--dry_run` (log sensor timestamps)
   - Re-train with larger motor delay range
   - Re-train with motor torque-tracking error as additional DR

For both failure modes, the diagnostic is to add `--suspended` and watch
what the policy does when there's no contact. If it executes a clean
motion in the air, the policy is fine and the issue is contact-related.
If it doesn't, the observation pipeline has a bug.

---

## 9. Reference

- Trained policy: [`Shish999/cr7-freekick-g1`](https://huggingface.co/Shish999/cr7-freekick-g1)
  on Hugging Face
- Motion file path on Spark:
  `~/rayane/humanoid-freekick/third_party/HumanoidSoccer/motions/cr7-only/football_stylized-001_right.npz`
- Training config: `Tracking-Flat-G1-SoccerDestination-RNN-DR-v0` task
- Sim2sim script: `scripts/sim2sim_mujoco.py`
- The key bug fix that enabled sim2sim was the `base_ang_vel` double-rotation
  — MuJoCo `qvel[3:6]` for free joints is already in body frame, so do not
  apply `quat_inv(q)` to it. The hardware script applies the same
  convention (IMU gyroscope reading is body-frame).
