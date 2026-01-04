#!/bin/bash
# 录制 Mode1 Task1 视频并播放
# Task1: 0.3m/s 往前 4s，然后原地跳 3s

cd /home/abc/Hopper/Hopper-mujoco

echo "=== 录制 Mode1 Task1 ==="
echo "任务: 0.3m/s 往前 4s + 原地跳 3s"

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

# 启用 IMU+EKF：默认喂给控制器（更接近真机）
sim.use_estimator = True
sim.estimator_kind = 'ekf'
sim.estimator_feed_control = False

sim.reset()

# 使用优化后的参数（最大高度≈1.0m）
sim.controller.k = 1500     # 弹簧刚度（降低高度）
sim.controller.b = 45       # 阻尼（降低高度）
sim.controller.h = 0.03     # 目标跳跃高度（降低）
sim.controller.Kp = 1.0     # 能量补偿（稳定优先）
sim.controller.Kv = 0.08    # 速度前馈
sim.controller.Kr = 0.012   # 速度校正
sim.controller.Kpp_x = 100  # 姿态比例
sim.controller.Kpp_y = 100
sim.controller.Kpd_x = 16.0 # 姿态微分（优化）
sim.controller.Kpd_y = 16.0
sim.controller.hipTorqueLim = 20

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
output_path = 'videos/task1/task1_final.mp4'
writer = imageio.get_writer(output_path, fps=fps)

total_time = 7.0  # 4s前进 + 3s原地
forward_time = 4.0
forward_vel = 0.3  # m/s

frame_count = 0
bounce_count = 0
last_phase = None
vel_sum = 0.0
vel_samples = 0

while sim.sim_time < total_time:
    # 前4秒: 0.3m/s 前进，之后原地跳
    if sim.sim_time < forward_time:
        desired_vel = np.array([forward_vel, 0.0])
        phase_name = 'Forward'
    else:
        desired_vel = np.array([0.0, 0.0])
        phase_name = 'In-place'
    
    for _ in range(steps_per_frame):
        state, torque, info = sim.step(desired_vel)
        if state['body_pos'][2] < 0.15:
            break
        
        # 计数跳跃次数
        current_phase = info.get('phase', None)
        if last_phase == 2 and current_phase == 1:  # stance -> flight
            bounce_count += 1
        last_phase = current_phase
        
        # 统计前进速度
        if sim.sim_time < forward_time:
            vel_sum += state['body_vel'][0]
            vel_samples += 1
    
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
    
    # 显示当前阶段和目标
    if phase_name == 'Forward':
        color = (0, 255, 255)  # 黄色
        text = f'Mode1 Task1: Forward @ {forward_vel} m/s'
    else:
        color = (0, 255, 0)  # 绿色
        text = 'Mode1 Task1: In-place Hopping'
    cv2.putText(frame_bgr, text, (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    
    # 显示跳跃次数
    cv2.putText(frame_bgr, f'Bounces: {bounce_count}', (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    writer.append_data(frame_rgb)
    frame_count += 1

writer.close()

avg_vel = vel_sum / vel_samples if vel_samples > 0 else 0
print(f'')
print(f'=== Task1 结果 ===')
print(f'总时间: {sim.sim_time:.2f}s (目标: {total_time}s)')
print(f'跳跃次数: {bounce_count}')
print(f'前4秒平均速度: {avg_vel:.3f} m/s (目标: {forward_vel} m/s)')
print(f'')
print(f'录制完成！{frame_count} 帧')
print(f'视频保存到: {output_path}')
"

echo ""
echo "=== 播放视频 ==="
ffplay -autoexit videos/task1/task1_final.mp4 2>/dev/null

