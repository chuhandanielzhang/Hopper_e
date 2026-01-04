# 🚁 Hopper + Propeller (Hybrid Hopper) Location

## 📍 Source Code Location

The **Hybrid Hopper** (Hopper + Propeller) reinforcement learning code is located at:

```
/home/abc/walk_these_ways_learning/walk-these-ways/go1_gym/envs/hybrid_hopper/
```

### Key Files:

1. **`hybrid_hopper_env.py`** - Main environment implementation
   - Inherits from `LeggedRobot` (same as Go1)
   - 7 DOF: 3 leg (Roll, Pitch, Shift) + 4 rotors (FL, FR, RL, RR)
   - Rotor thrusts applied as external forces

2. **`hybrid_hopper_config.py`** - Configuration
   - Stiffness: Roll=20, Pitch=20, Shift=200
   - Damping: Roll=0.5, Pitch=0.5, Shift=2
   - Action scale: [0.5, 0.5, 10] for [Roll, Pitch, Shift]
   - Rotor max thrust: 5.0 N per rotor

3. **URDF Model**:
   ```
   /home/abc/walk_these_ways_learning/walk-these-ways/resources/robots/hybrid_hopper/urdf/hybrid_hopper.urdf
   ```

## 🎮 Training Scripts

Located in `/home/abc/walk_these_ways_learning/`:

- **`start_hopper_training.sh`** - Start training
- **`play_trained_hopper_gui.sh`** - Play trained policy
- **`visualize_hybrid_hopper_gui.sh`** - Visualize hybrid hopper with GUI

## 📊 Training Logs

Training logs are in this folder (`Hopper_rl_t-master/logs/`), but note:
- Current logs are for **standard Hopper (leg-only)**, not hybrid hopper
- Hybrid hopper training would generate logs in the same structure

## 🔄 Difference: Standard vs Hybrid

### Standard Hopper (Leg-only):
- **Location**: `/home/abc/walk_these_ways_learning/walk-these-ways/go1_gym/envs/hopper/`
- **DOF**: 3 (Roll, Pitch, Shift)
- **No rotors**: Pure leg dynamics

### Hybrid Hopper (With Propellers):
- **Location**: `/home/abc/walk_these_ways_learning/walk-these-ways/go1_gym/envs/hybrid_hopper/`
- **DOF**: 7 (3 leg + 4 rotors)
- **Rotors**: FL, FR, RL, RR for attitude control
- **Rotor control**: Automatic PD control (not policy-controlled)

## 📹 Videos

Best training videos are copied to:
- `Hopper_rl_t-master/videos/best/hopper_leg_only_final.mp4` (latest checkpoint)
- `Hopper_rl_t-master/videos/best/hopper_leg_only_5000it.mp4` (5000 iterations)

**Note**: These are for standard Hopper. Hybrid hopper videos would be in similar locations if training was run.


