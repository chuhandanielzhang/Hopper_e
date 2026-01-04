#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="$(cd .. && pwd)/videos"
mkdir -p "$OUT_DIR"

OUT_MP4="$OUT_DIR/hopper4_leg_task_3s0_5s_fwd_3s0.mp4"

echo "=== Hopper_sim / model_spring: record Hopper4 LEG-only demo (3s in-place, 5s forward, 3s in-place) ==="
echo "Output: $OUT_MP4"

# Clean up old processes (best-effort)
pkill -f mujoco_lcm_fake_robot.py 2>/dev/null || true
pkill -f run_hopper4_leg_sim.py 2>/dev/null || true
pkill -f Hopper4.py 2>/dev/null || true
sleep 1

# Start MuJoCo fake robot (serial plant; same MJCF as ModeE)
# --hold-level-s 3.0: hold robot level for 3s before releasing (let controller start)
python3 ../model_aero/mujoco_lcm_fake_robot.py \
  --arm \
  --realtime \
  --model "$(cd .. && pwd)/mjcf/hopper_serial.xml" \
  --q-sign 1 \
  --q-offset 0 \
  --hold-level-s 3.0 \
  --duration-s 14 \
  --record-mp4 "$OUT_MP4" \
  --hud \
  > /tmp/hopper_sim_hopper4_leg_mj.log 2>&1 &
MJ_PID=$!

sleep 0.3

# Start Hopper4 controller (leg only). Let the launcher stop it (avoids wall-time drift vs sim-time).
python3 run_hopper4_leg_sim.py --duration-s 0 --vx-fwd 0.20 > /tmp/hopper_sim_hopper4_leg_ctl.log 2>&1 &
CTL_PID=$!

echo "Running... (MuJoCo PID=$MJ_PID, controller PID=$CTL_PID)"
wait "$MJ_PID" || true

echo "Stopping controller..."
kill "$CTL_PID" 2>/dev/null || true
wait "$CTL_PID" 2>/dev/null || true

if [ -f "$OUT_MP4" ]; then
  echo "✅ Done: $OUT_MP4"
  ls -lh "$OUT_MP4"
else
  echo "❌ Video not found: $OUT_MP4"
  echo "--- tail mujoco log ---"
  tail -80 /tmp/hopper_sim_hopper4_leg_mj.log 2>/dev/null || true
  echo "--- tail controller log ---"
  tail -80 /tmp/hopper_sim_hopper4_leg_ctl.log 2>/dev/null || true
  exit 1
fi


