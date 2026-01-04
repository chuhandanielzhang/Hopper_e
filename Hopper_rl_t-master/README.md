# Hopper RL Training (Walk These Ways)

This folder contains reinforcement learning training code and logs for the Hopper robot, based on the **Walk These Ways** framework.

## 📁 Folder Structure

```
Hopper_rl_t-master/
├── hopper_gym/                    # Gymnasium environment definitions
│   └── envs/
│       ├── hopper/               # Standard Hopper (leg-only, no propellers)
│       └── [other robots]        # A1, Go1, Mini Cheetah, etc.
│
├── rsl_rl/                       # RSL-RL training algorithms
│
└── logs/                         # Training logs and videos
    ├── hopper/                   # Standard Hopper training runs
    │   └── Nov25_13-59-16_/     # Latest training run
    │       ├── model_*.pt        # Checkpoint files
    │       └── videos/           # Training videos (15MB)
    │
    └── hopper_rl_5000it/         # 5000-iteration training run
        └── Nov23_19-35-05_/
            ├── model_5000.pt     # Final checkpoint
            └── videos/           # Training videos (15MB)
```

## 🚁 Hopper + Propeller (Hybrid Hopper)

**Location**: `/home/abc/walk_these_ways_learning/walk-these-ways/go1_gym/envs/hybrid_hopper/`

The **Hybrid Hopper** environment includes:
- **3 DOF leg**: Roll, Pitch, Shift (spring)
- **4 rotors**: FL, FR, RL, RR for attitude control
- **Total 7 DOF**: 3 leg + 4 rotors

### Key Files:
- `hybrid_hopper_env.py` - Environment implementation
- `hybrid_hopper_config.py` - Configuration (stiffness, damping, rewards)
- URDF: `walk-these-ways/resources/robots/hybrid_hopper/urdf/hybrid_hopper.urdf`

### Training Scripts:
- `/home/abc/walk_these_ways_learning/start_hopper_training.sh` - Start training
- `/home/abc/walk_these_ways_learning/play_trained_hopper_gui.sh` - Play trained policy
- `/home/abc/walk_these_ways_learning/visualize_hybrid_hopper_gui.sh` - Visualize hybrid hopper

## 🦵 Standard Hopper (Leg-only)

**Location**: `/home/abc/walk_these_ways_learning/walk-these-ways/go1_gym/envs/hopper/`

The **Standard Hopper** environment (no propellers):
- **3 DOF leg**: Roll, Pitch, Shift (spring)
- **No rotors**: Pure leg dynamics

### Key Files:
- `hopper_env.py` - Environment implementation
- `hopper_config.py` - Configuration
- URDF: `walk-these-ways/resources/robots/hopper/urdf/hopper.urdf`

## 📹 Training Videos

### Latest Training Run (Nov25_13-59-16_)
- **Location**: `logs/hopper/Nov25_13-59-16_/videos/`
- **Best video**: `04650.mp4` or `04800.mp4` (latest checkpoints)
- **Total size**: ~15MB

### 5000-iteration Run (Nov23_19-35-05_)
- **Location**: `logs/hopper_rl_5000it/Nov23_19-35-05_/videos/`
- **Final checkpoint**: `model_5000.pt`
- **Total size**: ~15MB

## 🎯 Quick Start

### Train Standard Hopper:
```bash
cd /home/abc/walk_these_ways_learning
bash start_hopper_training.sh
```

### Play Trained Policy:
```bash
cd /home/abc/walk_these_ways_learning
bash play_trained_hopper_gui.sh
```

### Visualize Hybrid Hopper:
```bash
cd /home/abc/walk_these_ways_learning
bash visualize_hybrid_hopper_gui.sh
```

## 📊 Training Configuration

From `logs/hopper/Nov25_13-59-16_/env_config.txt`:
- **Control**: Jacobian-based, PD control
- **Stiffness**: Roll=20, Pitch=20, Shift=200
- **Damping**: Roll=0.5, Pitch=0.5, Shift=2
- **Action scale**: [0.5, 0.5, 10] for [Roll, Pitch, Shift]
- **Episode length**: 20 seconds
- **Num envs**: 4096

## 🔗 Related Projects

- **Walk These Ways**: Original framework at `/home/abc/walk_these_ways_learning/`
- **Hopper-aero**: Real robot control (ModeE controller)
- **Hopper_sim**: MuJoCo simulation models


