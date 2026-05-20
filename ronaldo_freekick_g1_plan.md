# Ronaldo Free Kick Imitation on Unitree G1 — Project Plan

## Goal

Replicate Cristiano Ronaldo's free kick motion on a **Unitree G1 humanoid robot** using a purely learned RL policy. The robot must perform a realistic 3–4 step run-up and kick a stationary ball, imitating Ronaldo's signature technique as closely as possible.

The ball is placed at a **fixed, known position** relative to the robot's feet — no camera or spatial perception is needed. The full pipeline goes from **video pose extraction → motion retargeting → RL training in simulation → sim-to-sim validation → sim-to-real deployment**.

---

## Stack

| Component | Tool |
|---|---|
| Pose extraction | WHAM |
| Body model | SMPL |
| Retargeting | GMR (General Motion Retargeting) |
| Simulator | IsaacLab / IsaacSim |
| RL algorithm | PPO |
| Sim-to-sim | MuJoCo via GMR ROS node |
| Robot | Unitree G1 |
| Recording | DJI Spark |

---

## Pipeline Overview

```
YouTube video (side-on Ronaldo free kick)
        ↓
    [Phase 1] Pose Extraction — WHAM
        ↓ SMPL .pkl (world-grounded joint angles + root trajectory)
    [Phase 2] Motion Retargeting — GMR
        ↓ G1 joint angle trajectory (30 fps .pkl)
    [Phase 3] RL Training — IsaacLab + PPO
        ↓ Trained policy checkpoint
    [Phase 4] Sim-to-Sim Validation — MuJoCo
        ↓ Validated policy
    [Phase 5] Sim-to-Real Deployment — Unitree G1
        ↓ Working real-world demo
```

---

## Phase 1 — Pose Extraction

### Goal
Extract a world-grounded 3D SMPL motion sequence from a monocular YouTube video of Ronaldo performing a free kick.

### Why WHAM
- Outputs **world-grounded** motion (full root trajectory in 3D space), not just camera-relative pose
- This is critical: the robot physically takes 3–4 steps forward, so root translation must be accurate
- Handles broadcast/moving cameras well
- MIT license, simple to run
- Widely used in humanoid robotics pipelines

### Video selection criteria
- Camera angle: **strictly side-on** (90° to run-up direction)
- Full body visible head-to-toe throughout the entire motion
- Camera relatively static (no heavy panning/zooming)
- Shows complete sequence: **starting stance → 3–4 step run-up → plant foot → kick contact**
- Slow-motion clips preferred (less motion blur on the kicking leg)
- Search: `"Ronaldo free kick slow motion side view"`

### Steps

**1. Install WHAM**
```bash
git clone https://github.com/yohanshin/WHAM.git
cd WHAM
conda create -n wham python=3.10
conda activate wham
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**2. Download SMPL body model**
- Register at https://smpl.is.tue.mpg.de (free, non-commercial)
- Download `SMPL_NEUTRAL.pkl`
- Place in `WHAM/body_models/smpl/`

**3. Download WHAM pretrained checkpoint**
```bash
bash fetch_demo_data.sh
```

**4. Run WHAM on your video**
```bash
python demo.py \
  --video path/to/ronaldo_freekick.mp4 \
  --output_dir ./output \
  --save_pkl \
  --visualize
```

**5. Inspect and trim the output**
```python
import pickle
import numpy as np

with open('output/wham_output.pkl', 'rb') as f:
    data = pickle.load(f)

# Keys: poses, betas, trans, joints, ...
print(data.keys())
print("Total frames:", data['poses'].shape[0])

# Trim to the relevant segment (run-up + kick)
# Find frame indices visually from the --visualize output
start_frame = X   # first step of run-up
end_frame   = Y   # frame after ball contact

trimmed = {k: v[start_frame:end_frame] for k, v in data.items() 
           if isinstance(v, np.ndarray)}
```

**6. Smooth the trajectory**
```python
from scipy.signal import savgol_filter

# Apply Savitzky-Golay filter to remove jitter
poses_smooth = savgol_filter(trimmed['poses'], 
                              window_length=7, polyorder=3, axis=0)
trans_smooth  = savgol_filter(trimmed['trans'],  
                              window_length=7, polyorder=3, axis=0)
```

**7. Normalize root position**
```python
# Set starting root position to origin
trans_smooth -= trans_smooth[0]
```

**8. Save cleaned reference motion**
```python
trimmed['poses'] = poses_smooth
trimmed['trans'] = trans_smooth

with open('reference_motion.pkl', 'wb') as f:
    pickle.dump(trimmed, f)
```

### Output
`reference_motion.pkl` — SMPL pose parameters (θ) + root trajectory for the complete free kick motion.

---

## Phase 2 — Motion Retargeting

### Goal
Convert the SMPL human skeleton joint angles to Unitree G1 URDF joint angles, producing a physically valid reference trajectory for the robot.

### Why GMR
- Directly compatible with SMPL format input
- Explicitly supports Unitree G1 retargeting
- Comes with a built-in MuJoCo ROS node for sim-to-sim validation
- Produces joint trajectories that respect the G1's joint limits

### Steps

**1. Install GMR**
```bash
git clone https://github.com/retargeting-matters/GMR.git
cd GMR
conda create -n gmr python=3.10
conda activate gmr
pip install -r requirements.txt
```

**2. Configure G1 URDF**
- Point GMR to the Unitree G1 URDF (available in IsaacLab assets or Unitree SDK)
- Define key body correspondences: pelvis, hips, knees, ankles, shoulders, elbows

**3. Run retargeting**
```bash
python retarget.py \
  --input reference_motion.pkl \
  --robot g1 \
  --output g1_reference_motion.pkl \
  --fps 30
```

**4. Validate visually**
- Render the retargeted motion in MuJoCo
- Check: no self-collisions, joint limits respected, kick leg reaches ball height

**5. Post-process**
- Manually verify the foot contact frame (plant foot + kicking foot)
- Note the exact frame index of ball contact → used for sparse kick reward

### Output
`g1_reference_motion.pkl` — G1 joint angle trajectory at 30fps, ready for IsaacLab.

---

## Phase 3 — RL Training in IsaacLab

### Goal
Train a PPO policy in IsaacLab that tracks the reference motion while making actual contact with the ball.

### Environment Design

**Observation space** (what the policy sees at each timestep):
```
- Joint positions         (23 DOF)
- Joint velocities        (23 DOF)
- Root linear velocity    (3)
- Root angular velocity   (3)
- Root orientation        (4, quaternion)
- Phase clock             (2, sin/cos of motion phase)
- Ball position relative to feet (3)
- Reference joint positions at current phase (23 DOF)
```
Total: ~84 dimensions

**Action space:**
- PD target joint angles (23 DOF)
- PD gains: kp=100, kd=5 (tune per joint group)

**Episode structure:**
- Start: robot in neutral standing pose, ball at fixed offset from feet
- End: ball contact detected OR timeout (length of reference motion + buffer) OR robot falls

### Reward Function

```python
def compute_reward(obs, ref, phase):

    # 1. Motion tracking (dominant term)
    joint_error = torch.norm(obs.joint_pos - ref.joint_pos[phase], dim=-1)
    R_motion = torch.exp(-5.0 * joint_error)

    # 2. Root trajectory tracking
    root_pos_error = torch.norm(obs.root_pos - ref.root_pos[phase], dim=-1)
    root_vel_error = torch.norm(obs.root_vel - ref.root_vel[phase], dim=-1)
    R_root = torch.exp(-2.0 * root_pos_error) * torch.exp(-1.0 * root_vel_error)

    # 3. Sparse ball contact reward
    foot_ball_dist = torch.norm(obs.kick_foot_pos - obs.ball_pos, dim=-1)
    R_ball = torch.where(foot_ball_dist < 0.05, 
                         torch.tensor(1.0), 
                         torch.tensor(0.0))

    # 4. Stability penalty
    R_stability = -1.0 * (obs.fell.float())           # fell over
    R_torque    = -0.001 * torch.norm(obs.torques, dim=-1)  # large torques

    # 5. Feet contact pattern
    ref_contact = ref.foot_contact[phase]
    R_contact = torch.exp(-10.0 * 
                torch.norm(obs.foot_contact.float() - ref_contact.float(), dim=-1))

    # Weighted sum
    R_total = (0.60 * R_motion +
               0.15 * R_root   +
               0.10 * R_ball   +
               0.10 * R_stability + R_torque +
               0.05 * R_contact)

    return R_total
```

### Training Configuration

```python
# PPO hyperparameters (starting point)
num_envs        = 4096        # parallel environments
num_steps       = 24          # rollout steps per update
learning_rate   = 3e-4
gamma           = 0.99
gae_lambda      = 0.95
clip_range      = 0.2
num_epochs      = 5
batch_size      = num_envs * num_steps
entropy_coeff   = 0.01
max_grad_norm   = 1.0

# Domain randomization
joint_friction_range  = [0.8, 1.2]   # multiplier
link_mass_range       = [0.9, 1.1]   # multiplier
floor_friction_range  = [0.5, 1.0]
push_interval         = 200           # steps between random pushes
push_force            = 50            # Newtons
```

### Training procedure

1. **Stage 1 — Motion tracking only** (first 500M steps)
   - Set `w_ball = 0`, focus purely on learning the motion
   - Policy should learn to run-up and swing leg in Ronaldo style

2. **Stage 2 — Add ball contact reward** (next 200M steps)
   - Enable `w_ball = 0.10`
   - Policy refines the kick to actually contact the ball

3. **Monitoring**
   - Track: episode return, motion tracking error, ball contact rate, fall rate
   - Use IsaacLab's native TensorBoard logging

### Output
Trained PPO policy checkpoint: `checkpoint_best.pt`

---

## Phase 4 — Sim-to-Sim Validation

### Goal
Validate the policy in a second physics engine (MuJoCo) before touching the real robot. If it works across two different simulators, it is robust enough for real-world deployment.

### Steps

**1. Export policy to TorchScript**
```python
scripted = torch.jit.script(policy)
scripted.save('policy_scripted.pt')
```

**2. Set up GMR MuJoCo ROS node**
- The GMR repo includes a ROS node running MuJoCo for exactly this purpose
- Load the G1 URDF in MuJoCo
- Connect the policy as the controller

**3. Run 100 rollouts**
```bash
python sim2sim_eval.py \
  --policy policy_scripted.pt \
  --robot g1 \
  --num_rollouts 100 \
  --render
```

**4. Evaluate**

| Metric | Target |
|---|---|
| Ball contact rate | > 80% |
| Fall rate | < 5% |
| Motion tracking error | < 0.15 rad mean |
| Episode completion | > 90% |

**5. If metrics are not met:**
- Increase domain randomization range and retrain
- Check for sim-to-sim gap in specific joints (often ankles, hips)
- Add joint-specific observation noise during IsaacLab training

---

## Phase 5 — Sim-to-Real Deployment

### Goal
Deploy the validated policy on the physical Unitree G1 and record a clean demo with the DJI Spark.

### Hardware setup
- G1 in a flat, clear area with sufficient run-up space (~2m)
- Ball placed at the exact same offset used during training
- DJI Spark positioned side-on at hip height, static hover

### Deployment steps

**1. Connect to G1**
```bash
# Via Unitree SDK
source /opt/unitree/setup.bash
python deploy.py --policy policy_scripted.pt --robot g1
```

**2. Safety protocol**
- First run: policy at **30% speed** with a human ready to catch/stop
- Gradually increase to 60%, 80%, 100% speed across sessions
- Always have the emergency stop ready

**3. Observation pipeline on hardware**
```
G1 state estimator → joint pos/vel, root state
        ↓
Policy inference (20ms cycle, 50Hz)
        ↓
PD controller → motor torque commands
```

**4. Tuning if needed**
- If the kick is slightly off: adjust ball position offset by ±2cm
- If unstable: reduce PD gains by 10–15%
- If motion looks wrong: check that observation normalization matches training stats

**5. Record demo**
- DJI Spark: side-on, static hover, hip height
- Record 5–10 attempts, pick the cleanest one

---

## Key References

| Paper | Relevance |
|---|---|
| WHAM (Shin et al., 2024) | Monocular video → world-grounded SMPL |
| GMR (2025) | SMPL → Unitree G1 retargeting + sim2sim |
| ASAP (2025) | Full pipeline: video → SMPL → G1 RL → real robot |
| PPO (Schulman et al., 2017) | RL algorithm |
| AMP (Peng et al., 2021) | Reference motion imitation for physics-based control |
| IsaacLab | Training simulator |

---

## Timeline

| Phase | Task | Estimated Time |
|---|---|---|
| 1 | Video selection + WHAM extraction + cleaning | 1–2 days |
| 2 | GMR retargeting + visual validation | 1–2 days |
| 3 | IsaacLab env setup + PPO training | 1–2 weeks |
| 4 | Sim-to-sim validation + fixes | 2–3 days |
| 5 | Sim-to-real deployment + demo recording | 2–3 days |
