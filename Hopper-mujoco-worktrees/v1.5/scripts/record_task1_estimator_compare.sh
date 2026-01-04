#!/bin/bash
# 录制 Task1：左侧显示真实(REAL)，右侧显示估计(EST)
# Task1: 0.3m/s 往前 4s，然后原地跳 3s

set -euo pipefail

cd /home/abc/Hopper/Hopper-mujoco

mkdir -p videos/estimator

echo "=== 录制 Task1 (REAL vs EST) ==="
echo "输出: videos/estimator/task1_real_vs_est.mp4"

python3 -c "
import numpy as np
import cv2
import imageio
import mujoco
from scipy.spatial.transform import Rotation as R

from scripts.run_raibert_mj import HopperSimulation

sim = HopperSimulation(mode='leg', verbose=False)
sim.domain_rand_enabled = False
sim.use_estimator = True
sim.estimator_feed_control = False  # 只对比，不让估计器影响控制稳定性
sim.reset()

# 使用当前代码参数（高度≈1.0m 版本）
# Task1 里为了姿态更稳，脚本不强行改 controller 参数（以代码为准）

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
output_path = 'videos/estimator/task1_real_vs_est.mp4'
writer = imageio.get_writer(output_path, fps=fps)

total_time = 7.0
forward_time = 4.0
forward_vel = 0.3

err_vx = []
err_vy = []

def fmt_vec2(v):
    return f\"({v[0]:+.3f}, {v[1]:+.3f})\"

while sim.sim_time < total_time:
    desired_vel = np.array([forward_vel, 0.0]) if sim.sim_time < forward_time else np.array([0.0, 0.0])

    for _ in range(steps_per_frame):
        state, torque, info = sim.step(desired_vel)
        if state['body_pos'][2] < 0.15:
            break

    if state['body_pos'][2] < 0.15:
        print(f'摔倒 @ t={sim.sim_time:.2f}s')
        break

    # camera follow
    cam.lookat[0] = state['body_pos'][0]
    cam.lookat[1] = state['body_pos'][1]

    renderer.update_scene(sim.data, camera=cam, scene_option=opt)
    frame = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # read true/est
    v_true = state.get('true_body_vel', state['body_vel'])
    v_est = state.get('est_body_vel', state['body_vel'])
    q_true = state.get('true_body_quat', state['body_quat'])
    q_est = state.get('est_body_quat', state['body_quat'])

    rpy_true = R.from_quat([q_true[1], q_true[2], q_true[3], q_true[0]]).as_euler('xyz', degrees=True)
    rpy_est  = R.from_quat([q_est[1],  q_est[2],  q_est[3],  q_est[0]]).as_euler('xyz', degrees=True)

    err_vx.append(float(v_est[0] - v_true[0]))
    err_vy.append(float(v_est[1] - v_true[1]))

    # overlay layout
    xL, xR = 20, 660
    y0 = 40
    dy = 40

    cv2.putText(frame_bgr, f'Time: {sim.sim_time:.2f}s', (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.putText(frame_bgr, 'REAL', (xL, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    cv2.putText(frame_bgr, 'EST',  (xR, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.putText(frame_bgr, f'v_xy: {fmt_vec2(v_true)} m/s', (xL, y0 + 2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame_bgr, f'v_xy: {fmt_vec2(v_est)} m/s',  (xR, y0 + 2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame_bgr, f'Pitch: {rpy_true[1]:+.1f} deg', (xL, y0 + 3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame_bgr, f'Pitch: {rpy_est[1]:+.1f} deg',  (xR, y0 + 3*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame_bgr, f'Height: {state[\"body_pos\"][2]:.2f} m', (20, y0 + 4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # error stats (running)
    rmse_vx = (np.mean(np.square(err_vx)) ** 0.5) if err_vx else 0.0
    rmse_vy = (np.mean(np.square(err_vy)) ** 0.5) if err_vy else 0.0
    cv2.putText(frame_bgr, f'RMSE(vx,vy) = ({rmse_vx:.3f}, {rmse_vy:.3f}) m/s', (20, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # phase text
    phase_name = 'Forward' if sim.sim_time < forward_time else 'In-place'
    color = (0,255,255) if phase_name == 'Forward' else (0,255,0)
    cv2.putText(frame_bgr, f'Task1 Phase: {phase_name}', (20, y0 + 5*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

writer.close()
rmse_vx = (np.mean(np.square(err_vx)) ** 0.5) if err_vx else 0.0
rmse_vy = (np.mean(np.square(err_vy)) ** 0.5) if err_vy else 0.0
print('=== 估计误差统计 ===')
print(f'RMSE vx: {rmse_vx:.4f} m/s')
print(f'RMSE vy: {rmse_vy:.4f} m/s')
print(f'视频保存到: {output_path}')
"

echo ""
echo "=== 播放视频 ==="
ffplay -autoexit videos/estimator/task1_real_vs_est.mp4 2>/dev/null


