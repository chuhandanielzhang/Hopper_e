#!/bin/bash
# 录制 Mode1 Task0 原地跳视频并播放

cd /home/abc/Hopper/Hopper-mujoco

echo "=== 录制 Mode1 Task0 原地跳 ==="

python -c "
import numpy as np
import cv2
import imageio
import mujoco
from scripts.run_raibert_mj import HopperSimulation

print('开始录制...')
sim = HopperSimulation(mode='leg', verbose=False)

# 录制时禁用 Domain Randomization 以获得稳定视频
sim.domain_rand_enabled = False

# 同时启用速度估计器：用于显示 EST 速度（但不影响控制）
sim.use_estimator = True
sim.estimator_feed_control = False

sim.reset()

# 使用优化后的参数（最大高度≈1.0m）
sim.controller.k = 1500    # 弹簧刚度（降低高度）
sim.controller.b = 45      # 阻尼（降低高度）
sim.controller.h = 0.03    # 目标跳跃高度（降低）
sim.controller.Kp = 1.0    # 能量补偿（降低）
sim.controller.Kv = 0.1    # 速度反馈（抵消漂移）
sim.controller.Kr = 0.0    # 无期望速度
sim.controller.Kpd_x = 10.0  # 原地跳用较低的微分增益
sim.controller.Kpd_y = 10.0

width, height = 1280, 720
renderer = mujoco.Renderer(sim.model, width, height)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.azimuth = 90
cam.elevation = -20
cam.distance = 4.0
cam.lookat[:] = [0.0, 0.0, 0.8]
opt = mujoco.MjvOption()

fps = 50
steps_per_frame = 20
output_path = 'videos/task0/task0_final.mp4'
writer = imageio.get_writer(output_path, fps=fps)

total_time = 6.0
frame_count = 0

while sim.sim_time < total_time:
    for _ in range(steps_per_frame):
        state, torque, info = sim.step(np.array([0.0, 0.0]))
        if state['body_pos'][2] < 0.15:
            break
    
    if state['body_pos'][2] < 0.15:
        print(f'摔倒 @ t={sim.sim_time:.2f}s')
        break
    
    cam.lookat[0] = state['body_pos'][0]
    cam.lookat[1] = state['body_pos'][1]
    
    renderer.update_scene(sim.data, camera=cam, scene_option=opt)
    frame = renderer.render()
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 显示时间
    cv2.putText(frame_bgr, f'Time: {sim.sim_time:.2f}s', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 显示速度：REAL + EST（filter）
    vel_true = state.get('true_body_vel', state['body_vel'])
    vel_est = state.get('est_body_vel', vel_true)
    cv2.putText(frame_bgr, f'REAL Vel X: {vel_true[0]:.2f} m/s', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f'REAL Vel Y: {vel_true[1]:.2f} m/s', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f'EST  Vel X: {vel_est[0]:.2f} m/s', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200, 255, 200), 2)
    cv2.putText(frame_bgr, f'EST  Vel Y: {vel_est[1]:.2f} m/s', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200, 255, 200), 2)
    
    # 显示高度和姿态
    cv2.putText(frame_bgr, f'Height: {state[\"body_pos\"][2]:.2f}m', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    rpy = state['body_rpy']
    cv2.putText(frame_bgr, f'Pitch: {np.degrees(rpy[1]):.1f} deg', (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    
    # 显示任务名称
    cv2.putText(frame_bgr, 'Mode1 Task0: In-place Hopping', (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    writer.append_data(frame_rgb)
    frame_count += 1

writer.close()
print(f'录制完成！{frame_count} 帧')
print(f'视频保存到: {output_path}')
"

echo ""
echo "=== 播放视频 ==="
ffplay -autoexit videos/task0/task0_final.mp4 2>/dev/null

