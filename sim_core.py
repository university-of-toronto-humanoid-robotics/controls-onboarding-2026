"""
Simulation plumbing shared by claw_machine.py and check_controller.py.
New members do not need to edit this file.
"""

import collections
import importlib.util
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent

ARM_JOINTS = ("shoulder", "elbow")                # controlled by ArmPID
GRIPPER_JOINTS = ("finger_left", "finger_right")  # controlled by GripperPID
JOINT_NAMES = ARM_JOINTS + GRIPPER_JOINTS
HOLE_HALF_WIDTH = 0.0225  # half the side length of the socket hole (m)
PEG_HALF_HEIGHT = 0.04


def scene_path(reference: bool = False):
    """scene.xml (yours) or reference/scene.xml (the mentors' solution)."""
    return ROOT / ("reference/scene.xml" if reference else "scene.xml")


def load_controller_module(reference: bool = False):
    """Load controller.py (yours) or reference/controller.py (the mentors' solution)."""
    path = ROOT / ("reference/controller.py" if reference else "controller.py")
    spec = importlib.util.spec_from_file_location("user_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(reference: bool = False):
    """Load the scene (with the student's Task 1 actuators) and check them."""
    model = mujoco.MjModel.from_xml_path(str(scene_path(reference)))
    act_ids, issues = check_actuators(model)
    return model, act_ids, issues


def check_actuators(model):
    """Find the actuator driving each joint and check that it is an ideal torque source."""
    act_ids, issues = {}, []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        found = [i for i in range(model.nu)
                 if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                 and model.actuator_trnid[i, 0] == jid]
        if not found:
            issues.append(f"Task 1: no actuator on joint '{name}'")
            act_ids[name] = None
            continue
        if len(found) > 1:
            issues.append(f"Task 1: {len(found)} actuators on joint '{name}', expected 1")
        i = found[0]
        act_ids[name] = i
        ideal = (model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_NONE
                 and model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
                 and model.actuator_gainprm[i, 0] == 1.0
                 and model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
                 and model.actuator_gear[i, 0] == 1.0)
        if not ideal:
            issues.append(f"Task 1: actuator on '{name}' is not an ideal torque source")
        if not model.actuator_ctrllimited[i]:
            issues.append(f"Task 1: actuator on '{name}' has no ctrlrange")
    return act_ids, issues


class JointIndex:
    """qpos/qvel addresses and limits for the controlled joints."""

    def __init__(self, model):
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.range = np.array([model.jnt_range[j] for j in jids])

    def q(self, data):
        return data.qpos[self.qadr].copy()

    def qdot(self, data):
        return data.qvel[self.vadr].copy()


class Gantry:
    """The WASD-driven arm base: integrates a velocity command into the gantry servo targets."""

    def __init__(self, model):
        self.act = np.array([model.actuator(n).id for n in ("gantry_x", "gantry_y")])
        self.qadr = np.array([model.joint(n).qposadr[0] for n in ("gantry_x", "gantry_y")])
        self.range = model.actuator_ctrlrange[self.act]
        self.target = np.zeros(2)

    def reset(self, data):
        self.target = data.qpos[self.qadr].copy()
        data.ctrl[self.act] = self.target

    def move(self, data, vel_xy, dt):
        self.target = np.clip(self.target + np.asarray(vel_xy) * dt, self.range[:, 0], self.range[:, 1])
        data.ctrl[self.act] = self.target


class Servo:
    """Runs the student's PID for every joint and writes the torques to data.ctrl."""

    def __init__(self, model, ctrl_module, mode, act_ids):
        self.mode = mode
        self.joints = JointIndex(model)
        self.act_ids = act_ids
        self.dt = model.opt.timestep
        gains = ctrl_module.POSITION_GAINS if mode == "position" else ctrl_module.VELOCITY_GAINS
        self.classes = [ctrl_module.ArmPID if n in ARM_JOINTS else ctrl_module.GripperPID
                        for n in JOINT_NAMES]
        self.pids = [cls(*gains[n], ctrl_module.TORQUE_LIMITS[n])
                     for cls, n in zip(self.classes, JOINT_NAMES)]
        self.tau = np.zeros(len(JOINT_NAMES))
        self.error = None  # set if the student's PID is not implemented yet

    def reset(self):
        self.tau[:] = 0
        for pid in self.pids:
            try:
                pid.reset()
            except NotImplementedError:
                self.error = f"Task 2: {type(pid).__name__}.reset() not implemented"

    def apply(self, data, target):
        """target: desired positions (position mode) or velocities (velocity mode), one per joint."""
        q, qdot = self.joints.q(data), self.joints.qdot(data)
        for i, pid in enumerate(self.pids):
            try:
                if self.mode == "position":
                    tau = pid.position_torque(q[i], qdot[i], target[i], self.dt)
                else:
                    tau = pid.velocity_torque(qdot[i], target[i], self.dt)
            except NotImplementedError:
                self.error = f"Task 2: {type(pid).__name__}.{self.mode}_torque() not implemented"
                tau = 0.0
            self.tau[i] = 0.0 if tau is None else float(tau)
        for i, name in enumerate(JOINT_NAMES):
            if self.act_ids[name] is not None:
                data.ctrl[self.act_ids[name]] = self.tau[i]


class Teleop:
    """The human operator's command channel (Task 3).

    The operator nudges a *requested* setpoint in fixed increments of epsilon
    (the dead-band). Once every `period` seconds the current request is sampled
    and sent. It reaches the joint controller `delay` seconds later.

    Commands have three channels: shoulder, elbow, gripper. The gripper command
    goes to both fingers.
    """

    SHOULDER, ELBOW, GRIPPER = range(3)

    def __init__(self, mode, eps, period, delay, vel_limit, joint_range):
        self.mode = mode
        self.eps = np.asarray(eps, dtype=float)
        self.period = period
        self.delay = delay
        self.vel_limit = np.asarray(vel_limit, dtype=float)
        # The gripper channel uses finger_left's range (both fingers are identical).
        self.pos_range = joint_range[:3]
        self.steps = np.zeros(3, dtype=int)  # request = origin + steps * eps
        self.origin = np.zeros(3)

    def reset(self, q_home, t=0.0):
        self.origin = q_home[:3].copy() if self.mode == "position" else np.zeros(3)
        self.steps[:] = 0
        self.applied = self.request()
        self.queue = collections.deque()
        self.next_tick = t

    def request(self):
        return self.origin + self.steps * self.eps

    def nudge(self, channel, direction):
        self.steps[channel] += direction
        r = self.request()[channel]
        if self.mode == "position":
            lo, hi = self.pos_range[channel]
        else:
            lo, hi = -self.vel_limit[channel], self.vel_limit[channel]
        if not lo - 1e-9 <= r <= hi + 1e-9:
            self.steps[channel] -= direction

    def stop(self):
        """Velocity mode only: request zero velocity on every channel."""
        if self.mode == "velocity":
            self.steps[:] = 0

    def update(self, t):
        """Call every physics step. Returns the per-joint target the controller should track."""
        if t + 1e-9 >= self.next_tick:
            self.queue.append((self.next_tick + self.delay, self.request()))
            self.next_tick += self.period
        while self.queue and self.queue[0][0] <= t + 1e-9:
            self.applied = self.queue.popleft()[1]
        return self.expand(self.applied)

    @staticmethod
    def expand(cmd):
        return np.array([cmd[0], cmd[1], cmd[2], cmd[2]])


def peg_inserted(model, data):
    peg = data.body("peg").xpos
    hole = data.site("hole").xpos
    dx, dy = np.abs(peg[:2] - hole[:2])
    # The hole is 7 cm deep and the peg is 8 cm tall. Count it as inserted when
    # the peg's center is inside the hole footprint and below the rim.
    return dx < HOLE_HALF_WIDTH and dy < HOLE_HALF_WIDTH and peg[2] < hole[2] - 0.01


def grip_force(model, data):
    """Total normal contact force between the fingers and the peg (N)."""
    fingers = {model.geom("finger_left").id, model.geom("finger_right").id}
    peg = model.geom("peg").id
    total, f = 0.0, np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} & fingers and peg in (c.geom1, c.geom2):
            mujoco.mj_contactForce(model, data, i, f)
            total += abs(f[0])
    return total
