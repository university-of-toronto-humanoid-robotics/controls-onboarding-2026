"""
UTHR controls onboarding: THIS IS YOUR FILE.

Task 1 is done in scene.xml (the <actuator> section). Task 2 is done here:
fill in the functions marked TODO.

    Task 2  ArmPID      shoulder and elbow
            GripperPID  finger_left and finger_right

            For each class, fill in:
              position_torque()    Desired position -> torque.
              velocity_torque()    Desired velocity -> torque.
              reset()              Clear the controller's internal state.

Joints in the model (see scene.xml):
    "shoulder"      hinge, rad, rotates about world y-axis
    "elbow"         hinge, rad, rotates about world y-axis
    "finger_left"   slide, m,   positive = open
    "finger_right"  slide, m,   positive = open

Tune the gains below by editing this file and re-running the program.
Gains are not set with command-line flags.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Gains. Change these and re-run.
# Hinge joints: torque in N*m. Slide joints (fingers): force in N.
# ---------------------------------------------------------------------------

#                   kp      ki     kd
POSITION_GAINS = {
    "shoulder":     (20.0,  0.0,   1.0),
    "elbow":        (10.0,  0.0,   0.5),
    "finger_left":  (200.0, 0.0,   2.0),
    "finger_right": (200.0, 0.0,   2.0),
}

#                   kp      ki     kd
VELOCITY_GAINS = {
    "shoulder":     (3.0,   0.0,   0.0),
    "elbow":        (1.5,   0.0,   0.0),
    "finger_left":  (50.0,  0.0,   0.0),
    "finger_right": (50.0,  0.0,   0.0),
}

# Saturation limit for each joint's motor (N*m for hinges, N for slides).
# Keep these equal to the ctrlrange of your actuators in scene.xml.
TORQUE_LIMITS = {
    "shoulder":     10.0,
    "elbow":        6.0,
    "finger_left":  10.0,
    "finger_right": 10.0,
}


# ---------------------------------------------------------------------------
# Task 2: two PID controllers, one for the arm and one for the gripper
# ---------------------------------------------------------------------------

class ArmPID:
    """PID for the arm joints (hinges, output in N*m).

    The simulator creates one ArmPID instance for EACH of "shoulder" and "elbow",
    using that joint's gains for the selected mode (POSITION_GAINS or
    VELOCITY_GAINS). It calls exactly one of position_torque() or
    velocity_torque() every physics step (dt = 0.002 s), and it calls reset()
    when the scene is reset.
    """

    def __init__(self, kp: float, ki: float, kd: float, limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        # TODO (Task 2): add any internal state you need (integral, previous error, ...)
        #   - A PID needs memory between calls: at least a running integral of
        #     the error, and the previous error if you differentiate the error.
        #   - Make sure that state exists before the first control call
        #     (e.g. initialise it here, or just call self.reset()).

    def reset(self) -> None:
        """Clear the internal state (integral, previous error, ...)."""
        # TODO (Task 2)
        #   - Set the integral back to zero.
        #   - Forget the previous error, so the first call after a reset does
        #     not compute a huge derivative from stale data (hint: a sentinel
        #     like None lets you detect "no previous error yet").
        #   - Reset anything else you added in __init__.
        raise NotImplementedError

    def position_torque(self, q: float, qdot: float, q_des: float, dt: float) -> float:
        """Position mode. Return the torque/force that drives the joint position q to q_des.

        q     measured joint position (rad or m)
        qdot  measured joint velocity (rad/s or m/s)
        q_des desired joint position
        dt    time since the last call (s)
        """
        # TODO (Task 2)
        #   1. Error:      e = q_des - q   (check the sign: wrong sign = positive feedback)
        #   2. Integral:   accumulate e * dt into your stored integral.
        #      Anti-windup: stop the integral from growing without bound when
        #      the motor saturates (e.g. clamp it, or only integrate while the
        #      output is not saturated).
        #   3. Derivative: q_des jumps in steps during Task 3, so d(e)/dt spikes
        #      ("derivative kick"). Between steps q_des is constant, so
        #      de/dt = -qdot. Consider using the measured qdot instead.
        #   4. Torque:     tau = kp*P + ki*I + kd*D
        #   5. Saturate:   clamp tau to [-self.limit, +self.limit] and return it
        #      as a plain float.
        raise NotImplementedError

    def velocity_torque(self, qdot: float, qdot_des: float, dt: float) -> float:
        """Velocity mode. Return the torque/force that drives the joint velocity qdot to qdot_des.

        qdot     measured joint velocity (rad/s or m/s)
        qdot_des desired joint velocity
        dt       time since the last call (s)
        """
        # TODO (Task 2)
        #   1. Error:      e = qdot_des - qdot
        #   2. Integral:   accumulate e * dt, with anti-windup (same idea as in
        #      position mode). Think about what the integral of a velocity
        #      error means physically, and why it fights gravity drift.
        #   3. Derivative: rate of change of the error, (e - previous e) / dt.
        #      Handle the first call (no previous error yet), then store e
        #      for the next call.
        #   4. Torque:     tau = kp*P + ki*I + kd*D
        #   5. Saturate:   clamp tau to [-self.limit, +self.limit] and return it
        #      as a plain float.
        raise NotImplementedError


class GripperPID:
    """PID for the gripper joints (slides, output in N).

    The simulator creates one GripperPID instance for EACH of "finger_left" and "finger_right",
    using that joint's gains for the selected mode (POSITION_GAINS or
    VELOCITY_GAINS). It calls exactly one of position_torque() or
    velocity_torque() every physics step (dt = 0.002 s), and it calls reset()
    when the scene is reset.

    Same structure as ArmPID, but the job is different: the fingers squeeze a
    peg they cannot close through, so the error never reaches zero while
    gripping. Think about what that does to the integral term (windup) and how
    hard you want the fingers to squeeze. You may reuse ArmPID's law or write
    a different one.
    """

    def __init__(self, kp: float, ki: float, kd: float, limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        # TODO (Task 2): add any internal state you need (integral, previous error, ...)
        #   - A PID needs memory between calls: at least a running integral of
        #     the error, and the previous error if you differentiate the error.
        #   - Make sure that state exists before the first control call
        #     (e.g. initialise it here, or just call self.reset()).

    def reset(self) -> None:
        """Clear the internal state (integral, previous error, ...)."""
        # TODO (Task 2)
        #   - Set the integral back to zero.
        #   - Forget the previous error, so the first call after a reset does
        #     not compute a huge derivative from stale data (hint: a sentinel
        #     like None lets you detect "no previous error yet").
        #   - Reset anything else you added in __init__.
        raise NotImplementedError

    def position_torque(self, q: float, qdot: float, q_des: float, dt: float) -> float:
        """Position mode. Return the torque/force that drives the joint position q to q_des.

        q     measured joint position (rad or m)
        qdot  measured joint velocity (rad/s or m/s)
        q_des desired joint position
        dt    time since the last call (s)
        """
        # TODO (Task 2)
        #   1. Error:      e = q_des - q   (check the sign: wrong sign = positive feedback)
        #   2. Integral:   accumulate e * dt into your stored integral.
        #      Anti-windup: stop the integral from growing without bound when
        #      the motor saturates (e.g. clamp it, or only integrate while the
        #      output is not saturated).
        #   3. Derivative: q_des jumps in steps during Task 3, so d(e)/dt spikes
        #      ("derivative kick"). Between steps q_des is constant, so
        #      de/dt = -qdot. Consider using the measured qdot instead.
        #   4. Torque:     tau = kp*P + ki*I + kd*D
        #   5. Saturate:   clamp tau to [-self.limit, +self.limit] and return it
        #      as a plain float.
        raise NotImplementedError

    def velocity_torque(self, qdot: float, qdot_des: float, dt: float) -> float:
        """Velocity mode. Return the torque/force that drives the joint velocity qdot to qdot_des.

        qdot     measured joint velocity (rad/s or m/s)
        qdot_des desired joint velocity
        dt       time since the last call (s)
        """
        # TODO (Task 2)
        #   1. Error:      e = qdot_des - qdot
        #   2. Integral:   accumulate e * dt, with anti-windup (same idea as in
        #      position mode). Think about what the integral of a velocity
        #      error means physically, and why it fights gravity drift.
        #   3. Derivative: rate of change of the error, (e - previous e) / dt.
        #      Handle the first call (no previous error yet), then store e
        #      for the next call.
        #   4. Torque:     tau = kp*P + ki*I + kd*D
        #   5. Saturate:   clamp tau to [-self.limit, +self.limit] and return it
        #      as a plain float.
        raise NotImplementedError
