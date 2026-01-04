# 📹 Hopper RL Training Videos

## 🎬 最佳训练视频（已整理）

### 本地文件路径：

1. **最新训练最终 checkpoint** (Nov25_13-59-16_, iteration 4950):
   ```
   /home/abc/Hopper/Hopper_rl_t-master/videos/best/hopper_leg_only_final.mp4
   ```
   相对路径：`Hopper_rl_t-master/videos/best/hopper_leg_only_final.mp4`

2. **5000迭代训练最终 checkpoint** (Nov23_19-35-05_, iteration 4950):
   ```
   /home/abc/Hopper/Hopper_rl_t-master/videos/best/hopper_leg_only_5000it.mp4
   ```
   相对路径：`Hopper_rl_t-master/videos/best/hopper_leg_only_5000it.mp4`

---

## 📂 完整训练视频库

### 最新训练 (Nov25_13-59-16_)

**位置**：`Hopper_rl_t-master/logs/hopper/Nov25_13-59-16_/videos/`

**视频数量**：198 个视频（每个 checkpoint 一个）

**最佳视频**：
- `04950.mp4` - 最终 checkpoint (143KB)
- `04900.mp4` - 倒数第二个 (152KB)
- `04850.mp4` - 倒数第三个 (152KB)

**完整路径示例**：
```
/home/abc/Hopper/Hopper_rl_t-master/logs/hopper/Nov25_13-59-16_/videos/04950.mp4
```

### 5000迭代训练 (Nov23_19-35-05_)

**位置**：`Hopper_rl_t-master/logs/hopper_rl_5000it/Nov23_19-35-05_/videos/`

**最佳视频**：
- `04950.mp4` - 最终 checkpoint (143KB)
- `04900.mp4` - 倒数第二个 (152KB)
- `04850.mp4` - 倒数第三个 (152KB)

**完整路径示例**：
```
/home/abc/Hopper/Hopper_rl_t-master/logs/hopper_rl_5000it/Nov23_19-35-05_/videos/04950.mp4
```

---

## 🖥️ 如何查看视频

### 方法 1: 使用文件管理器
```bash
# 打开文件管理器，导航到：
cd /home/abc/Hopper/Hopper_rl_t-master/videos/best/
# 双击 .mp4 文件即可播放
```

### 方法 2: 使用命令行播放器
```bash
# 使用 VLC
vlc /home/abc/Hopper/Hopper_rl_t-master/videos/best/hopper_leg_only_final.mp4

# 或使用 mpv
mpv /home/abc/Hopper/Hopper_rl_t-master/videos/best/hopper_leg_only_final.mp4

# 或使用 ffplay
ffplay /home/abc/Hopper/Hopper_rl_t-master/videos/best/hopper_leg_only_final.mp4
```

### 方法 3: 在 Python 中显示
```python
import subprocess
subprocess.run(['vlc', '/home/abc/Hopper/Hopper_rl_t-master/videos/best/hopper_leg_only_final.mp4'])
```

---

## 📊 视频信息

- **格式**: MP4 (H.264)
- **大小**: ~140-150KB per video
- **内容**: 训练过程中的机器人行为记录
- **时长**: 每个视频约 20 秒（一个 episode）

---

## 🔗 如果需要在线链接

如果要将视频上传到 GitHub 或其他平台：

1. **GitHub**: 可以 push 到仓库，然后使用 raw.githubusercontent.com 链接
2. **YouTube**: 上传后获得分享链接
3. **其他平台**: 根据平台要求上传

**注意**: 当前视频文件在本地，需要手动上传才能获得在线链接。


