"""
Headless check of Tasks 1 and 2. No window opens, there is no delay, and no dead-band.

    python check_controller.py --position
    python check_controller.py --velocity

Position mode: holds the home pose for 1 s, steps every joint to a new target,
then reports overshoot, settling time and steady-state error.

Velocity mode: commands zero velocity for 1 s (how far does gravity drag the
arm?), then a constant velocity for 2 s, and reports the tracking error.

Re-run after every gain change in controller.py to compare tunings.
"""

import argparse

import mujoco
import numpy as np

import sim_core
from sim_core import JOINT_NAMES

POSITION_STEP = np.array([+0.4, -0.4, -0.02, -0.02])  # added to the home pose
VELOCITY_CMD = np.array([+0.2, -0.2, 0.0, 0.0])


def simulate(model, servo, targets_fn, duration):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    servo.reset()
    home = servo.joints.q(data)
    log_t, log_q, log_qdot = [], [], []
    while data.time < duration:
        servo.apply(data, targets_fn(data.time, home))
        mujoco.mj_step(model, data)
        log_t.append(data.time)
        log_q.append(servo.joints.q(data))
        log_qdot.append(servo.joints.qdot(data))
    return home, np.array(log_t), np.array(log_q), np.array(log_qdot)


def check_position(model, servo):
    t_step = 1.0
    target = lambda t, home: home + (POSITION_STEP if t >= t_step else 0)
    home, t, q, _ = simulate(model, servo, target, t_step + 3.0)
    after = t >= t_step
    print(f"Step at t = {t_step:.1f} s, then 3 s of response.\n")
    print(f"{'joint':<14}{'step':>9}{'overshoot':>12}{'settle(5%)':>12}{'ss error':>12}{'hold err':>11}")
    for i, name in enumerate(JOINT_NAMES):
        step = POSITION_STEP[i]
        goal = home[i] + step
        resp = (q[after, i] - home[i]) / step  # normalized: 0 -> 1
        overshoot = max(0.0, resp.max() - 1.0) * 100
        outside = np.nonzero(np.abs(resp - 1.0) > 0.05)[0]
        settle = (t[after][outside[-1]] - t_step) if len(outside) else 0.0
        settle_txt = f"{settle:.2f} s" if len(outside) < after.sum() else "never"
        ss = goal - q[-1, i]
        hold = home[i] - q[np.searchsorted(t, t_step) - 1, i]
        print(f"{name:<14}{step:>+9.3f}{overshoot:>11.1f}%{settle_txt:>12}{ss:>+12.4f}{hold:>+11.4f}")
    print("\nhold err = error just before the step (gravity load at the home pose).")


def check_velocity(model, servo):
    t_cmd = 1.0
    target = lambda t, home: VELOCITY_CMD if t >= t_cmd else np.zeros(4)
    home, t, q, qdot = simulate(model, servo, target, t_cmd + 2.0)
    hold = t < t_cmd
    tail = t >= t_cmd + 1.0
    print(f"Zero velocity for {t_cmd:.1f} s, then the command below for 2 s.\n")
    print(f"{'joint':<14}{'drift @ 0 cmd':>15}{'cmd':>9}{'mean vel err':>15}")
    for i, name in enumerate(JOINT_NAMES):
        drift = q[hold, i][-1] - home[i]
        err = np.mean(VELOCITY_CMD[i] - qdot[tail, i])
        print(f"{name:<14}{drift:>+15.4f}{VELOCITY_CMD[i]:>+9.2f}{err:>+15.4f}")
    print("\ndrift = position change while commanding zero velocity.")
    print("mean vel err = average (desired - measured) over the last second.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--position", dest="mode", action="store_const", const="position")
    mode.add_argument("--velocity", dest="mode", action="store_const", const="velocity")
    parser.add_argument("--reference", action="store_true")
    parser.set_defaults(mode="position")
    args = parser.parse_args()

    ctrl = sim_core.load_controller_module(args.reference)
    model, act_ids, issues = sim_core.build_model(args.reference)
    if issues:
        print("Task 1 is not done yet:\n  " + "\n  ".join(issues))
        return
    print("Task 1 OK: every joint has an ideal torque-source actuator.\n")

    servo = sim_core.Servo(model, ctrl, args.mode, act_ids)
    gains = ctrl.POSITION_GAINS if args.mode == "position" else ctrl.VELOCITY_GAINS
    print(f"{args.mode.upper()} mode controllers (kp, ki, kd):")
    for cls, n in zip(servo.classes, JOINT_NAMES):
        print(f"  {n:<14}{cls.__name__:<14}{gains[n]}")
    print()
    (check_position if args.mode == "position" else check_velocity)(model, servo)
    if servo.error:
        print(f"\n{servo.error}: the arm got zero torque.")


if __name__ == "__main__":
    main()
