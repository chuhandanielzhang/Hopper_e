#!/bin/bash
# 录制 Mode2 Task1 视频并播放
# Task1: 0.3m/s 往前 4s，然后原地跳 3s
#
# 默认使用 leg_prop（腿+旋翼融合控制），也可通过环境变量切换：
#   MODE=leg bash scripts/record_mode2_task1.sh

set -euo pipefail

cd /home/abc/Hopper/Hopper-mujoco

MODE="${MODE:-leg_prop}"

echo "=== 录制 Mode2 Task1 ==="
echo "模式: ${MODE}"
echo "任务: 0.3m/s 往前 4s + 原地跳 3s"

python3 -c "
import numpy as np
import cv2
import imageio
import mujoco
from scripts.run_raibert_mj import HopperSimulation

mode = '${MODE}'
print('开始录制...')
sim = HopperSimulation(mode=mode, verbose=False, controller_mode=2)

# 录制时禁用 Domain Randomization 以获得稳定视频
sim.domain_rand_enabled = False

# 同时启用速度估计器：用于显示 EST 速度（但不影响控制）
sim.use_estimator = True
sim.estimator_feed_control = False
# 使用完整 EKF（ESKF）作为滤波器输出
sim.estimator_kind = 'ekf'

sim.reset()

# 录制参数（与当前高度≈1.0m配置一致，Task1 额外增强姿态阻尼）
sim.controller.k = 1500
sim.controller.b = 45
sim.controller.h = 0.03
sim.controller.Kp = 1.0
sim.controller.Kv = 0.08
sim.controller.Kr = 0.012
sim.controller.Kpp_x = 100
sim.controller.Kpp_y = 100
sim.controller.Kpd_x = 16.0
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
output_path = 'videos/task1/mode2_task1_ekf.mp4'
writer = imageio.get_writer(output_path, fps=fps)

total_time = 7.0
forward_time = 4.0
forward_vel = 0.3

frame_count = 0
bounce_count = 0
last_state = None
vel_sum = 0.0
vel_samples = 0

while sim.sim_time < total_time:
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

        # 计数跳跃次数（stance -> flight）
        cur_state = info.get('state', None)
        if last_state == 2 and cur_state == 1:
            bounce_count += 1
        last_state = cur_state

        # 统计前进速度（用 REAL）
        if sim.sim_time < forward_time:
            v_true = state.get('true_body_vel', state['body_vel'])
            vel_sum += v_true[0]
            vel_samples += 1

    if state['body_pos'][2] < 0.15:
        print(f'摔倒 @ t={sim.sim_time:.2f}s')
        break

    cam.lookat[0] = state['body_pos'][0]
    cam.lookat[1] = state['body_pos'][1]

    renderer.update_scene(sim.data, camera=cam, scene_option=opt)
    frame = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 叠加信息：REAL + EST
    v_true = state.get('true_body_vel', state['body_vel'])
    v_est = state.get('est_body_vel', v_true)

    cv2.putText(frame_bgr, f'Time: {sim.sim_time:.2f}s', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.putText(frame_bgr, f'REAL Vel X: {v_true[0]:.2f} m/s', (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,255,255), 2)
    cv2.putText(frame_bgr, f'REAL Vel Y: {v_true[1]:.2f} m/s', (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,255,255), 2)
    cv2.putText(frame_bgr, f'EST  Vel X: {v_est[0]:.2f} m/s', (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200,255,200), 2)
    cv2.putText(frame_bgr, f'EST  Vel Y: {v_est[1]:.2f} m/s', (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200,255,200), 2)

    cv2.putText(frame_bgr, f'Height: {state[\"body_pos\"][2]:.2f}m', (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,255,255), 2)
    rpy = state['body_rpy']
    cv2.putText(frame_bgr, f'Pitch: {np.degrees(rpy[1]):.1f} deg', (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,255,255), 2)

    # 显示 prop 信息（leg_prop 才有）
    if info.get('mode') == 'leg_prop':
        thrust = info.get('prop_thrust', 0.0)
        Mx = info.get('prop_L', 0.0)
        My = info.get('prop_M', 0.0)
        cv2.putText(frame_bgr, f'Prop: T={thrust:.2f}N Mx={Mx:.2f} My={My:.2f}', (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

    text = f'Mode2 Task1 ({mode}): {phase_name}'
    color = (0,255,255) if phase_name == 'Forward' else (0,255,0)
    cv2.putText(frame_bgr, text, (20, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

    cv2.putText(frame_bgr, f'Bounces: {bounce_count}', (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,255,255), 2)

    writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    frame_count += 1

writer.close()

avg_vel = vel_sum / vel_samples if vel_samples > 0 else 0.0
print('')
print('=== Mode2 Task1 结果 ===')
print(f'总时间: {sim.sim_time:.2f}s (目标: {total_time}s)')
print(f'跳跃次数: {bounce_count}')
print(f'前4秒平均速度(REAL): {avg_vel:.3f} m/s (目标: {forward_vel} m/s)')
print('')
print(f'录制完成！{frame_count} 帧')
print(f'视频保存到: {output_path}')
"

echo ""
echo "=== 播放视频 ==="
ffplay -autoexit videos/task1/mode2_task1_ekf.mp4 2>/dev/null



