# controls-onboarding-2026

UTHR controls onboarding: servomotor dynamics and control modes for dexterous manipulation in MuJoCo.

You will build the low-level motor control for a **claw machine**: a two-joint arm that hangs from a gantry and has a two-finger gripper. Then you will operate it by hand. You must pick up an orange peg and insert it into the hole in the green socket.

**What you are learning:** how manipulation performance changes between **position control** and **velocity control**, and between **different P gains**, when a human operator closes the outer loop through a slow, delayed and quantized command channel. The low-level PID is your servo. You, the operator, act as the adaptive controller around it.

---

## 1. Setup

You need Python 3.10 or newer.

```bash
git clone <this repo>
cd controls-onboarding-2026
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check that it runs:

```bash
python claw_machine.py --position
```

A window opens. The arm hangs limp and swings, and the top-right corner lists `TODO` items. That is expected, because nothing is implemented yet.

> On **macOS** use plain `python`, not `mjpython`. This project uses its own GLFW window, not MuJoCo's passive viewer.

---

## 2. What's in the repo

| File | Edit it? | Purpose |
|---|---|---|
| `scene.xml` | **YES (Task 1 only)** | MuJoCo model: gantry, arm, gripper, peg, socket. You add the arm actuators in its `<actuator>` section. |
| `controller.py` | **YES (Task 2)** | Empty `ArmPID` and `GripperPID` classes, plus the gains you tune. |
| `claw_machine.py` | No | Interactive simulator: the window, keyboard, and the delayed and quantized operator channel (Task 3). |
| `check_controller.py` | No | Headless test of Tasks 1 and 2: prints step-response numbers. |
| `sim_core.py` | No | Shared plumbing: model building, actuator checks, PID wiring, teleop channel. |
| `reference/` | — | Mentors' solution. See [For mentors](#7-for-mentors). |

### The robot

```
        ══════[ gantry ]══════        <- slides in x/y with WASD (provided, not your job)
                  ● shoulder           hinge about world y-axis  (rad)
                   \
                    \  link1 (0.40 m)
                     ● elbow           hinge about world y-axis  (rad)
                     |
                     |  link2 (0.30 m)
                   ┌─┴─┐ palm
                   │   │ finger_left / finger_right   slide joints (m), positive = OPEN, range 0 to 0.04
```

| Joint | Type | Units | Range | Motor limit (`TORQUE_LIMITS`) |
|---|---|---|---|---|
| `shoulder` | hinge | rad | ±2.0 | 10 N·m |
| `elbow` | hinge | rad | ±2.5 | 6 N·m |
| `finger_left` | slide | m | 0 to 0.04 | 10 N |
| `finger_right` | slide | m | 0 to 0.04 | 10 N |

Useful geometry (you will need it in Task 3):

- At `shoulder = elbow = 0` the arm hangs straight down and the fingertips are about 1 cm above the floor.
- The gripper points straight down when **`shoulder + elbow = 0`**.
- With the gripper pointing down, the fingertips sit at
  `x ≈ gantry_x − 0.4·sin(shoulder)` and `height ≈ 0.41 − 0.4·cos(shoulder)` (m).
  Making `shoulder` more negative raises the gripper **and** moves it toward +x, so you have to correct the x position with the gantry.
- The peg is 3 × 3 × 8 cm and weighs 50 g. The fingers touch it at about `finger = 0.01`.
- The hole is 4.5 × 4.5 cm and 7 cm deep, with its rim at 10 cm above the floor. The fingers are wider than the hole, so you lower the peg until it is just inside the rim and then let go.

---

## 3. Task 1: Ideal torque-source actuators

**Where:** `scene.xml`, in the `<actuator>` section near the bottom (look for `TODO (Task 1)`)

When you first open `scene.xml`, it only has the two gantry actuators. The arm and gripper have none, which is why the arm is limp. Add one **`<general>`** actuator per joint (`shoulder`, `elbow`, `finger_left`, `finger_right`) so that each joint becomes an **ideal torque (or force) source**: the generalized force on the joint equals the control signal, `τ = ctrl`.

In MuJoCo's general actuator model:

```
force = gain · (ctrl, or the activation if there are dynamics) + bias
gain  = gainprm[0]                                           when gaintype = "fixed"
bias  = biasprm[0] + biasprm[1]·length + biasprm[2]·velocity  when biastype = "affine"
```

Choose the attributes so that `force = ctrl` exactly:

- `joint`: the joint name. `name`: the same as the joint name (the simulator finds actuators by joint).
- **No activation dynamics**: ctrl acts instantly.
- **Fixed gain of 1.**
- **No bias**: no built-in spring or damper. Your PID does that job.
- **Limited control range** of `±` the motor limit: 10 N·m for the shoulder, 6 N·m for the elbow, 10 N for each finger. These must match `TORQUE_LIMITS` in `controller.py`.

Read the [`general` actuator reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general) to find the attribute names. Use `<general>`, not the `<motor>` shortcut, and write every attribute out explicitly, even when it matches the default.

**Done when:** `python check_controller.py` prints `Task 1 OK`. If an actuator is missing or isn't an ideal torque source, the checker (and the top-right of the sim window) says which one and why.

**Think about:** look at the two gantry `<position>` actuators right above yours. Which `<general>` attributes would turn your torque source into that P controller? What does a real servomotor's "position mode" do internally?

## 4. Task 2: PID from desired position / velocity to torque

**Where:** `controller.py` → classes `ArmPID` and `GripperPID`: `reset()`, `position_torque()`, `velocity_torque()`

The simulator calls each joint's controller **every physics step** (`dt = 0.002 s`):

| Mode | Call | Error | Torque |
|---|---|---|---|
| `--position` | `position_torque(q, qdot, q_des, dt)` | `e = q_des − q` | `τ = Kp·e + Ki·∫e dt + Kd·de/dt` |
| `--velocity` | `velocity_torque(qdot, qdot_des, dt)` | `e = q̇_des − q̇` | `τ = Kp·e + Ki·∫e dt + Kd·de/dt` |

Clamp the output to `±self.limit`. `reset()` must clear any state you keep, such as the integral and the previous error.

Gains are in `POSITION_GAINS` / `VELOCITY_GAINS` at the top of `controller.py`, written as `(kp, ki, kd)` for each joint. **Change gains by editing the file and re-running.** There are no command-line flags for gains.

**Two controllers.** You write two classes: `ArmPID` for the shoulder and elbow (hinges, output in N·m) and `GripperPID` for the two fingers (slides, output in N). Each joint gets its **own instance** of its class, so it has its own gains and its own integral and previous-error state. The gripper has a different job from the arm: it squeezes an object it can't move through, instead of tracking free motion. You may end up with a different law for each class.

Things to know:

1. **Derivative kick.** In Task 3 the setpoint jumps in steps. If you differentiate the error, what happens to `de/dt` at each step? In position mode, the setpoint is piecewise constant, so `de/dt = −q̇` between steps. Use that.
2. **Integral windup.** If the motor saturates (or the gripper is squeezing the peg and can't close), the integral keeps growing. What happens when you let go? Add an anti-windup scheme.
3. **Gravity.** With `Ki = 0`, what steady-state error does a P(D) controller have when the arm holds itself up against gravity? Measure it.

**Check it:**

```bash
python check_controller.py --position   # step response: overshoot, settling time, steady-state error
python check_controller.py --velocity   # drift while commanding 0 velocity, tracking error
```

Run it, change a gain, and run it again. Try at least three values of `Kp` for the shoulder (for example 5, 20, 80) in each mode and write down the numbers. You will use them in Task 3.

**Done when:** the arm holds its pose in the interactive sim, the position step response settles, and in velocity mode the drift under a zero-velocity command is small. You can explain how each gain changes the numbers.

---

## 5. Task 3: Be the controller

**Where:** `python claw_machine.py --position` or `python claw_machine.py --velocity`

Now you drive the arm by hand. The command link is deliberately bad, much like teleoperating a real robot over a slow link:

| Property | Value | Meaning |
|---|---|---|
| **Dead-band** (ε) | position: 0.05 rad / 0.005 m per key press; velocity: 0.05 rad/s / 0.01 m/s per key press | You can only change the setpoint in whole increments of ε. |
| **Big control steps** | 0.25 s | Your request is sampled only 4 times per second. |
| **Delay** | 0.30 s | Each sampled command reaches the motors 0.3 s later. |

Your PID from Task 2 still runs at 500 Hz underneath and tracks whatever setpoint arrives.

### Keys

| Key | Action |
|---|---|
| **W / S** | gantry +y / −y (away from / toward you) |
| **A / D** | gantry −x / +x (left / right) |
| **U / J** | shoulder setpoint + / − one ε |
| **I / K** | elbow setpoint + / − one ε |
| **O / L** | gripper open / close one ε (both fingers) |
| **Space** | velocity mode only: set every velocity setpoint back to 0 |
| **F** | **toggle contact-force arrows: use this to watch your grip force** |
| **C** | toggle contact points |
| **P** | pause |
| **R** | reset the scene and the timer |
| **H** | hide/show the key help |
| **Esc** | quit |
| Mouse | left-drag rotate, right-drag pan, scroll zoom (hold Shift for the horizontal variants) |

The on-screen panel shows each joint as `request > sent > measured | torque`:

- *request* is what you have asked for.
- *sent* is the delayed command the PID is currently tracking.
- *measured* is the actual joint value.
- *torque* is the PID output.

It also shows the **grip force** on the peg and a **task timer**. The timer starts on your first key press and stops when the peg is in the hole.

### The task

1. Raise the gripper and line it up over the peg. Keep `shoulder + elbow ≈ 0` so the gripper points down.
2. Lower it until the fingers are around the peg, then close the gripper. Press **F** and check that there are force arrows on both fingers.
3. Lift, carry the peg over the socket (W/S/A/D), and lower it until the peg's bottom is inside the hole.
4. Open the gripper. The panel shows `INSERTED in X s`.

### The experiment (this is the point of onboarding)

Run the experiment in different modes with different gains.

Questions to answer:

- Which mode was easier with the delay? Why? Think about what one key press means in each mode, and what happens if you stop pressing keys.
- How did a low `Kp` show up during the task (sag, a sloppy grip, lag)? How did a high `Kp` show up (oscillation, the peg being flung)?
- In velocity mode, how did you, the human, compensate for gravity drift? That compensation is you acting as the integral term.
- How did the delay change *your* strategy? Did you start making small moves and waiting?
- What would you change about the servo if you knew the operator had a 0.3 s delay?

---

## 6. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Arm hangs limp, `TODO` in the top-right | Task 1 or 2 is not implemented yet. The message tells you which. |
| `Task 1: actuator on 'x' is not an ideal torque source` | Check `dyntype`, `gaintype`, `gainprm`, `biastype`, and `gear` (must be 1) in `scene.xml`. |
| `XML Error` when starting | Typo in your `<general>` element; the message gives the line number. |
| Arm explodes or spins wildly | Wrong sign in your PID (positive feedback), or a huge `Kd` with `dt` in the wrong place. |
| Arm slowly sags in position mode | No integral term. That is expected behavior; explain it. |
| The peg slips out while moving the gantry | The gantry accelerates, so the arm swings. Squeeze harder (finger gains) or move gently. |
| Window doesn't open on Linux | `pip install glfw` needs a system OpenGL; try `sudo apt install libglfw3 libgl1`. |