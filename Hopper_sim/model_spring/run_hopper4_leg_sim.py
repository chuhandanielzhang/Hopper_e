#!/usr/bin/env python3
"""
Non-interactive Hopper4 (leg-only) runner for Hopper_sim.

Runs the Hopper4 LCM controller for a fixed duration and exits cleanly.
This avoids the interactive CLI + matplotlib plotting in Hopper4.py's `main()`.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from Hopper4 import HopperLCMController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=0.0, help="Run time (wall seconds). <=0 means run until killed.")
    ap.add_argument("--vx-fwd", type=float, default=0.30, help="Forward vx command during the middle segment (m/s).")
    ap.add_argument("--t-inplace-s", type=float, default=3.0, help="In-place duration at start/end (seconds).")
    ap.add_argument("--t-fwd-s", type=float, default=5.0, help="Forward duration in the middle (seconds).")
    args = ap.parse_args()

    ctl = HopperLCMController()

    # Velocity-mode flag (Hopper4 expects desired_velocity[2]=1.0)
    ctl.desired_velocity = np.array([0.0, 0.0, 1.0], dtype=float)

    # Leg-only: keep props disarmed
    ctl.propeller_armed = False
    ctl.propeller_vector_mode = False

    # LCM handler thread
    t_lcm = threading.Thread(target=ctl.run_lcm_handler, daemon=True)
    t_lcm.start()

    # Wall-clock start (for logging only)
    t0 = time.time()

    # Scripted demo: 3s in-place -> 5s forward -> 3s in-place (total 11s by default)
    t_in = float(max(0.0, args.t_inplace_s))
    t_fwd = float(max(0.0, args.t_fwd_s))
    vx_fwd = float(args.vx_fwd)

    def _cmd_scheduler() -> None:
        # Start: in-place
        ctl.desired_velocity = np.array([0.0, 0.0, 1.0], dtype=float)
        time.sleep(t_in)
        # Middle: forward
        ctl.desired_velocity = np.array([vx_fwd, 0.0, 1.0], dtype=float)
        time.sleep(t_fwd)
        # End: back to in-place
        ctl.desired_velocity = np.array([0.0, 0.0, 1.0], dtype=float)

    threading.Thread(target=_cmd_scheduler, daemon=True).start()

    # Optional stopper (wall time). For video recording we typically let the launcher script kill us.
    if float(args.duration_s) > 0.0:
        def _stopper() -> None:
            time.sleep(float(max(0.1, args.duration_s)))
            ctl.running = False

        threading.Thread(target=_stopper, daemon=True).start()

    try:
        ctl.run_controller()
    finally:
        ctl.running = False
        # Disarm props (in case anything armed them)
        try:
            for _ in range(5):
                ctl.send_motor_command(1000, 1000, 1000, False)
                time.sleep(0.01)
        except Exception:
            pass

    dt = time.time() - t0
    print(f"[hopper4_leg_sim] done (ran {dt:.2f}s)")


if __name__ == "__main__":
    main()


