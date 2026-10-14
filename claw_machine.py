"""
UTHR controls onboarding: the claw machine.

    python claw_machine.py --position     # you send desired joint POSITIONS
    python claw_machine.py --velocity     # you send desired joint VELOCITIES

Pick up the orange peg and insert it into the green socket. Press H in the
window to show the key bindings.
"""

import argparse
import time

import glfw
import mujoco
import numpy as np

import sim_core
from sim_core import JOINT_NAMES, Teleop

# ---------------------------------------------------------------------------
# Task 3 operator settings. Mentors may tune these. They are the "slow, delayed,
# quantized" command channel you control the arm through.
# ---------------------------------------------------------------------------
CONTROL_PERIOD = 0.2  # s: your command is sampled only once per control step
COMMAND_DELAY = 0.30   # s: each sampled command takes this long to reach the arm
#                       shoulder  elbow   gripper
EPS_POSITION = (0.05,    0.05,   0.005)  # dead-band: rad, rad, m per key press
EPS_VELOCITY = (0.05,    0.05,   0.01)   # dead-band: rad/s, rad/s, m/s per key press
MAX_VELOCITY = (1.0,     1.0,    0.05)

BASE_SPEED = 0.15                 # m/s for the WASD gantry

KEY_HELP = (
    "W / S\nA / D\nU / J\nI / K\nO / L\nSpace\nF\nC\nP\nR\nH\nEsc",
    "gantry +y / -y\ngantry -x / +x\nshoulder + / -\nelbow + / -\ngripper open / close\n"
    "zero all velocities (velocity mode)\ntoggle contact forces\ntoggle contact points\n"
    "pause\nreset\nhide this help\nquit",
)


class App:
    def __init__(self, mode, reference):
        self.mode = mode
        ctrl = sim_core.load_controller_module(reference)
        self.model, act_ids, self.issues = sim_core.build_model(reference)
        self.data = mujoco.MjData(self.model)
        self.home = self.model.key("home").id
        self.servo = sim_core.Servo(self.model, ctrl, mode, act_ids)
        self.gantry = sim_core.Gantry(self.model)
        joints = self.servo.joints
        eps = EPS_POSITION if mode == "position" else EPS_VELOCITY
        self.teleop = Teleop(mode, eps, CONTROL_PERIOD, COMMAND_DELAY, MAX_VELOCITY, joints.range)
        self.paused = False
        self.show_help = True
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home)
        mujoco.mj_forward(self.model, self.data)
        self.servo.reset()
        self.gantry.reset(self.data)
        self.teleop.reset(self.servo.joints.q(self.data))
        self.timer_start = None
        self.success_time = None

    # -- input -------------------------------------------------------------

    def on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        nudges = {
            glfw.KEY_U: (Teleop.SHOULDER, +1), glfw.KEY_J: (Teleop.SHOULDER, -1),
            glfw.KEY_I: (Teleop.ELBOW, +1), glfw.KEY_K: (Teleop.ELBOW, -1),
            glfw.KEY_O: (Teleop.GRIPPER, +1), glfw.KEY_L: (Teleop.GRIPPER, -1),
        }
        if key in nudges:
            self.teleop.nudge(*nudges[key])
            self.start_timer()
            return
        if action != glfw.PRESS:
            return
        flags = self.opt.flags
        if key == glfw.KEY_SPACE:
            self.teleop.stop()
        elif key == glfw.KEY_F:
            flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] ^= 1
        elif key == glfw.KEY_C:
            flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] ^= 1
        elif key == glfw.KEY_P:
            self.paused = not self.paused
        elif key == glfw.KEY_R:
            self.reset()
        elif key == glfw.KEY_H:
            self.show_help = not self.show_help
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def start_timer(self):
        if self.timer_start is None:
            self.timer_start = self.data.time

    def base_velocity(self, window):
        held = lambda k: glfw.get_key(window, k) == glfw.PRESS
        v = np.array([held(glfw.KEY_D) - held(glfw.KEY_A), held(glfw.KEY_W) - held(glfw.KEY_S)], dtype=float)
        if v.any():
            self.start_timer()
        return BASE_SPEED * v

    def on_mouse_button(self, window, button, action, mods):
        self.last_mouse = glfw.get_cursor_pos(window)

    def on_mouse_move(self, window, x, y):
        lx, ly = self.last_mouse
        self.last_mouse = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if not (left or right or middle):
            return
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                 or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        A = mujoco.mjtMouse
        if right:
            action = A.mjMOUSE_MOVE_H if shift else A.mjMOUSE_MOVE_V
        elif left:
            action = A.mjMOUSE_ROTATE_H if shift else A.mjMOUSE_ROTATE_V
        else:
            action = A.mjMOUSE_ZOOM
        _, height = glfw.get_window_size(window)
        mujoco.mjv_moveCamera(self.model, action, (x - lx) / height, (y - ly) / height,
                              self.scn, self.cam)

    def on_scroll(self, window, dx, dy):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * dy,
                              self.scn, self.cam)

    # -- simulation --------------------------------------------------------

    def step(self, base_vel):
        m, d = self.model, self.data
        self.gantry.move(d, base_vel, m.opt.timestep)
        target = self.teleop.update(d.time)
        self.servo.apply(d, target)
        mujoco.mj_step(m, d)
        if self.success_time is None and self.timer_start is not None \
                and sim_core.peg_inserted(m, d):
            self.success_time = d.time - self.timer_start

    # -- rendering ---------------------------------------------------------

    def overlay(self, viewport):
        d, font = self.data, mujoco.mjtFontScale.mjFONTSCALE_150
        G = mujoco.mjtGridPos
        q, qdot = self.servo.joints.q(d), self.servo.joints.qdot(d)
        req = Teleop.expand(self.teleop.request())
        app = Teleop.expand(self.teleop.applied)
        unit = "" if self.mode == "position" else "/s"
        names, values = ["mode", "", ""], [self.mode.upper(),
                                           f"step {CONTROL_PERIOD:.2f} s, delay {COMMAND_DELAY:.2f} s",
                                           "request > sent > measured | torque"]
        meas = q if self.mode == "position" else qdot
        for i, n in enumerate(JOINT_NAMES):
            names.append(n)
            values.append(f"{req[i]:+.3f} > {app[i]:+.3f} > {meas[i]:+.3f}{unit} | {self.servo.tau[i]:+.2f}")
        names += ["grip force", "contact force [F]", "task time"]
        on = self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]
        if self.success_time is not None:
            timer = f"INSERTED in {self.success_time:.1f} s  (R to retry)"
        elif self.timer_start is not None:
            timer = f"{d.time - self.timer_start:.1f} s"
        else:
            timer = "starts on first key press"
        values += [f"{sim_core.grip_force(self.model, d):.2f} N", "ON" if on else "off", timer]
        if self.paused:
            names.append("PAUSED")
            values.append("")
        mujoco.mjr_overlay(font, G.mjGRID_TOPLEFT, viewport, "\n".join(names), "\n".join(values), self.con)

        warnings = list(self.issues) + ([self.servo.error] if self.servo.error else [])
        if warnings:
            mujoco.mjr_overlay(font, G.mjGRID_TOPRIGHT, viewport,
                               "\n".join(["TODO"] * len(warnings)), "\n".join(warnings), self.con)
        if self.show_help:
            mujoco.mjr_overlay(font, G.mjGRID_BOTTOMLEFT, viewport, *KEY_HELP, self.con)

    def run(self):
        if not glfw.init():
            raise RuntimeError("could not initialize GLFW")
        window = glfw.create_window(1400, 900, f"UTHR claw machine ({self.mode} mode)", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (0, 0, 0.35)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.0, 90.0, -20.0
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.last_mouse = (0, 0)

        glfw.set_key_callback(window, self.on_key)
        glfw.set_mouse_button_callback(window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(window, self.on_mouse_move)
        glfw.set_scroll_callback(window, self.on_scroll)

        for issue in self.issues:
            print("WARNING:", issue)

        last = time.perf_counter()
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            frame_dt, last = min(now - last, 1 / 30), now  # cap so a hiccup doesn't explode
            if not self.paused:
                base_vel = self.base_velocity(window)
                t_end = self.data.time + frame_dt
                while self.data.time < t_end:
                    self.step(base_vel)

            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            mujoco.mjr_render(viewport, self.scn, self.con)
            self.overlay(viewport)
            glfw.swap_buffers(window)
            glfw.poll_events()

        self.con.free()
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--position", dest="mode", action="store_const", const="position",
                      help="command desired joint positions (default)")
    mode.add_argument("--velocity", dest="mode", action="store_const", const="velocity",
                      help="command desired joint velocities")
    parser.add_argument("--reference", action="store_true",
                        help="use reference/scene.xml and reference/controller.py (mentors)")
    parser.set_defaults(mode="position")
    args = parser.parse_args()
    App(args.mode, args.reference).run()


if __name__ == "__main__":
    main()
