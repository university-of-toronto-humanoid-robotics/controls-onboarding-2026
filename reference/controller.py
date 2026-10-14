"""
Reference solution for controller.py (Task 2). Task 1 is in reference/scene.xml.
MENTORS ONLY: do not give this to new members.

Run it with:  python claw_machine.py --reference
"""

import numpy as np

#                   kp      ki     kd
POSITION_GAINS = {
    "shoulder":     (40.0,  20.0,  2.0),
    "elbow":        (20.0,  10.0,  1.0),
    "finger_left":  (200.0, 0.0,   2.0),
    "finger_right": (200.0, 0.0,   2.0),
}

#                   kp      ki     kd
VELOCITY_GAINS = {
    "shoulder":     (3.0,   0.0,  0.0),
    "elbow":        (1.5,   0.0,  0.0),
    "finger_left":  (50.0,  0.0, 0.0),
    "finger_right": (50.0,  0.0, 0.0),
}

TORQUE_LIMITS = {
    "shoulder":     10.0,
    "elbow":        6.0,
    "finger_left":  10.0,
    "finger_right": 10.0,
}


class ArmPID:
    def __init__(self, kp: float, ki: float, kd: float, limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def _integrate(self, error: float, dt: float) -> None:
        self.integral += error * dt
        # Anti-windup: the I term alone may not exceed the actuator limit.
        if self.ki > 0:
            bound = self.limit / self.ki
            self.integral = float(np.clip(self.integral, -bound, bound))

    def position_torque(self, q: float, qdot: float, q_des: float, dt: float) -> float:
        error = q_des - q
        self._integrate(error, dt)
        # Derivative on measurement (-kd * qdot) instead of d(error)/dt, so the
        # step changes in q_des do not cause a "derivative kick".
        tau = self.kp * error + self.ki * self.integral - self.kd * qdot
        return float(np.clip(tau, -self.limit, self.limit))

    def velocity_torque(self, qdot: float, qdot_des: float, dt: float) -> float:
        error = qdot_des - qdot
        self._integrate(error, dt)
        d_error = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        tau = self.kp * error + self.ki * self.integral + self.kd * d_error
        return float(np.clip(tau, -self.limit, self.limit))


class GripperPID(ArmPID):
    """The same control law works for the fingers; only the gains differ."""
