# 📋 Hopper RL Training Summary

## 🎯 Quick Reference

### 🚁 Hopper + Propeller (Hybrid Hopper) - **强化学习代码位置**

**源代码位置**：
```
/home/abc/walk_these_ways_learning/walk-these-ways/go1_gym/envs/hybrid_hopper/
```

**关键文件**：
- `hybrid_hopper_env.py` - 环境实现（7 DOF: 3 leg + 4 rotors）
- `hybrid_hopper_config.py` - 配置（stiffness, damping, rewards）
- URDF: `walk-these-ways/resources/robots/hybrid_hopper/urdf/hybrid_hopper.urdf`

**训练脚本**：
- `/home/abc/walk_these_ways_learning/start_hopper_training.sh`
- `/home/abc/walk_these_ways_learning/play_trained_hopper_gui.sh`
- `/home/abc/walk_these_ways_learning/visualize_hybrid_hopper_gui.sh`

---

### 🦵 Standard Hopper (Leg-only) - **当前训练日志位置**

**源代码位置**：
```
/home/abc/walk_these_ways_learning/walk-these-ways/go1_gym/envs/hopper/
```

**训练日志位置**（本文件夹）：
```
Hopper_rl_t-master/logs/
├── hopper/Nov25_13-59-16_/          # 最新训练（~5000 iterations）
│   ├── model_*.pt                   # Checkpoint 文件
│   └── videos/*.mp4                 # 训练视频（198个视频，15MB）
│
└── hopper_rl_5000it/Nov23_19-35-05_/  # 5000迭代训练
    ├── model_5000.pt                # 最终 checkpoint
    └── videos/*.mp4                 # 训练视频（15MB）
```

---

## 📹 最佳训练视频

已复制到统一位置：
```
Hopper_rl_t-master/videos/best/
├── hopper_leg_only_final.mp4       # 最新训练最终 checkpoint (04950)
└── hopper_leg_only_5000it.mp4      # 5000迭代训练最终 checkpoint (04950)
```

**原始视频位置**：
- 最新训练：`logs/hopper/Nov25_13-59-16_/videos/04950.mp4`
- 5000迭代：`logs/hopper_rl_5000it/Nov23_19-35-05_/videos/04950.mp4`

---

## 📁 文件夹结构

```
Hopper_rl_t-master/
├── README.md                        # 主文档（英文）
├── HOPPER_PROP_LOCATION.md         # Hybrid Hopper 位置说明
├── SUMMARY.md                       # 本文件（中文总结）
│
├── hopper_gym/                     # Gymnasium 环境定义
│   └── envs/
│       ├── hopper/                 # Standard Hopper（只有 .pyc，源代码在 walk_these_ways_learning）
│       └── [其他机器人]
│
├── rsl_rl/                         # RSL-RL 训练算法
│
├── logs/                           # 训练日志和视频
│   ├── hopper/                     # Standard Hopper 训练
│   └── hopper_rl_5000it/           # 5000迭代训练
│
└── videos/                         # 整理后的最佳视频
    └── best/
        ├── hopper_leg_only_final.mp4
        └── hopper_leg_only_5000it.mp4
```

---

## 🔍 重要说明

1. **源代码不在本文件夹**：
   - 本文件夹 (`Hopper_rl_t-master`) 只包含**训练日志和视频**
   - 源代码在 `/home/abc/walk_these_ways_learning/walk-these-ways/go1_gym/envs/`

2. **Hybrid Hopper vs Standard Hopper**：
   - **Hybrid Hopper** = Hopper + 4 rotors (propellers)
   - **Standard Hopper** = 只有腿，没有螺旋桨
   - 当前训练日志是 **Standard Hopper** 的

3. **视频文件**：
   - 训练过程中自动生成，每个 checkpoint 一个视频
   - 最佳视频已复制到 `videos/best/` 目录

---

## 🚀 快速开始

### 查看训练视频：
```bash
cd /home/abc/Hopper/Hopper_rl_t-master
vlc videos/best/hopper_leg_only_final.mp4
```

### 训练 Hybrid Hopper：
```bash
cd /home/abc/walk_these_ways_learning
bash start_hopper_training.sh
```

### 播放训练好的策略：
```bash
cd /home/abc/walk_these_ways_learning
bash play_trained_hopper_gui.sh
```


