# MorphLab vision

MorphLab is a game-like robotics laboratory that runs entirely in the
browser. It teaches robotics by letting people build a robot, see the math
behind its motion, simulate it physically, tune its controllers, give it
senses, teach it skills and finally command those skills in plain language.

This document describes the full long-term vision. It is not a statement of
what exists. See `docs/status.md` for the implemented and verified state and
`docs/milestones.md` for the delivery order.

## Who it is for
- Students and self-learners who want to understand kinematics, dynamics,
  control, perception and learning by experiment rather than by formula only.
- Educators who need a zero-install, shareable demonstration environment.
- Hobbyists prototyping robot morphologies before building hardware.

## Capability areas

### 1. Build
Create a robot from links and joints. Choose the number of limbs and the
degrees of freedom per limb. Presets: robot arm, wheeled rover, quadruped,
hexapod. Link lengths, joint limits, masses and actuator strength are
editable. Symmetry tools copy a limb to its mirror positions. Every change to
the body creates a new morphology version.

### 2. Kinematics
Forward kinematics, inverse kinematics (damped least squares with joint
limits), Jacobian visualization (manipulability ellipsoid and singular
values), workspace sampling and reachability feedback. Joint axes and
coordinate frames are optional overlays.

### 3. Physics
Rigid-body simulation with gravity, contact, friction and joint actuators
using a real physics engine. Push and disturbance tools test robustness.

### 4. Control theory
PD joint control with live gain tuning, commanded-versus-measured plots,
step response metrics and visible actuator saturation. Educational overlays
explain what the controller is doing and why it may oscillate or sag.

### 5. Perception and navigation
A sensor-only mode, distinct from ground-truth debug mode: joint encoders,
IMU-like orientation, foot contacts, range rays and a virtual camera. A
wheeled rover builds an occupancy grid, plans a path and follows it. The
rover can be driven remotely by keyboard or run autonomously.

### 6. Learning
The user demonstrates or trains skills for a specific robot. A skill is
saved with metadata: morphology version, controller parameters or policy
weights, observation and action definitions, training configuration,
evaluation results and tested conditions. The progression is:
1. Record and replay (open-loop, labeled as replay).
2. Controller parameter tuning (closed-loop, hand-designed structure).
3. Imitation learning (a small policy fitted to demonstrations).
4. Reward-based improvement (evolution strategies on parameters or weights).
Each stage is labeled honestly in the interface.

### 7. Language
A command bar accepts text such as "lift your front-left leg and hold for
three seconds". A local rule-based parser converts the text into a
structured skill call, resolves names against the actual robot, checks the
saved skill library for a compatible and evaluated skill, executes it and
reports measured results. Unsupported requests receive a clear "cannot do
this yet" answer with a suggestion of what to teach first. An optional cloud
language model adapter may be configured by the user at runtime; its key is
never shipped in frontend code and never persisted by default.

### 8. Explainability
Show why a target is unreachable, why a controller oscillates, what the
sensors can and cannot observe, and the reward terms behind learned
behavior. A "compare two attempts" view replays two recorded attempts side
by side.

## Flagship demonstration
Build a quadruped, drag its feet with inverse kinematics, tune its stance
controller, improve a single-foot-lift skill, save it, reload the page and
trigger it with a text command.

## Boundaries
- Arbitrary bodies cannot learn arbitrary tasks. Skills are specific to a
  morphology and to the conditions they were evaluated under.
- A language model never generates real-time joint torques. Language is the
  slow, high-level interface; controllers and policies handle motion.
- Animation, physics, replay, control and learning are distinct modes and
  are never presented as one another.
- Simulation, control, learning and persistence run in the browser. No
  backend is required. A cloud language model is optional.
- Simulated sensors are idealized models. Results do not transfer to real
  hardware without further work, and the interface says so.
