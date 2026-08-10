# Research paths for language-driven robot skills

This note records the published work that shaped D25 (the AI composes,
checks and improves skills) and where each idea lives in MorphLab. Ideas
are reimplemented from the papers; no code was copied. Status of every
feature is in `docs/status.md`.

## Sources and what MorphLab takes from them

| Work | Idea | In MorphLab |
|---|---|---|
| Language to Rewards (Yu et al., 2023, arXiv 2306.08647) | A language model writes reward parameters instead of low-level actions; an optimizer (MuJoCo MPC) finds the motion. Reported success on 17 tasks, about 90 %. | The model writes goals (bounded reward terms judged from simulator ground truth) and a draft motion; evolution strategies improve the draft against the goal reward (`src/skills/improve.ts`). No MPC runs in the browser. |
| SayTap (Tang et al., 2023, arXiv 2306.07580) | Foot contact patterns (which foot is down in each phase of a cycle) as the interface between language and a locomotion controller. Gaits such as trot, pace, bound, standing on three legs; cycles near 0.5 s; contact ratios 0.5 to 0.7. | `src/skills/gait.ts`: a gait is a contact pattern (trot, walk, pace, bound, pronk, wave, stand, or per-leg phases), a duty factor, a cycle time, stride, side step, turn and step height. A trajectory generator and leg IK replace SayTap's learned policy. |
| Policies Modulating Trajectory Generators (Iscen et al., 2018, arXiv 1910.02812) | A linear feedback policy reading the IMU modulates a periodic trajectory generator; learned with ES in under 1,000 rollouts. | The gait's IMU feedback raises or lowers feet against body roll and pitch (gains `feedbackKp`, `feedbackKd`), and ES may tune those gains. The feedback is linear and hand-structured, not a learned network. |
| Eureka (Ma et al., 2023, arXiv 2310.12931) | Iterative reward design by a language model with reflection on training outcomes. | The model reads every measured goal after a try and revises the draft (at most three drafts per command); it may not weaken a goal to pass. Goals are checked against the request (below). |
| Evolution strategies (Salimans et al., 2017, arXiv 1703.03864) | Antithetic sampling, rank-based fitness shaping, common random numbers. | `src/learning/es.ts` (D3), reused with common randomized conditions per iteration and a held-out comparison before any change is kept. |
| SayCan (Ahn et al., 2022, arXiv 2204.01691) | Ground language plans in the robot's affordances: weight a skill by how likely it is to succeed here. | `runSkillSequence` estimates a sequence's chance of success from each skill's evaluated success rate before running it, and reports it next to the measured outcome. |
| Inner Monologue (Huang et al., 2022, arXiv 2207.05608) | Closed-loop language feedback: success detection and scene descriptions return to the model after each action. | Every tool result carries the measured outcome; the model must report from it (D14, D21). |
| Text2Motion (Lin et al., 2023, arXiv 2303.12153) | Check the feasibility of a skill sequence before executing it. | Sequences are validated (names, compatibility with the body, evaluation present) before the first skill runs. Geometric feasibility between skills is not checked; each skill restarts from the home stance. |
| MuJoCo MPC (Howell et al., 2022, arXiv 2212.00541; Apache-2.0) | Predictive sampling as a simple real-time planner. | Not used. Noted as a Tier C path for goals that keyframes and gaits cannot reach. |

## Checking goals against the request
SayTap and L2R both rely on the interface constraining what the model can
ask for. MorphLab adds a small, literal check (`src/skills/intent.ts`):
motion words in a skill's name or request map to a measurable property
(a sit pitches the body nose up, a wave swings a limb tip, walking moves
the body, a jump lifts every foot). A draft whose goals measure none of
them is refused with what to add. The lexicon is deliberately small;
words without a rule are not judged.

## Tier C (outside the browser)
- Reinforcement learning of locomotion policies (PPO in Isaac Lab or
  MuJoCo Playground) with domain randomization, exported to ONNX and bound
  to a morphology hash. Browser inference of small policies is feasible
  (D3); training at that scale is not.
- Model predictive control from language-written rewards (L2R with MJPC).
- Vision-language-action models: far too large for control-rate inference
  in the browser; at most a planner that selects MorphLab tools.

## Physical-AI repositories reviewed (2026-10-06, D26)
Nine repositories were read (README, documentation and, where noted, source
files) for what MorphLab's browser pipeline could use. All nine are Python
training frameworks; none can run in the browser, and no code was copied.
Licenses were read from each repository.

| Repository | License | Runs on | What it offers | Decision |
|---|---|---|---|---|
| leggedrobotics/rsl_rl | BSD-3-Clause | Python 3.9+, PyTorch, GPU, multi-GPU; used by Isaac Lab, Legged Gym, mjlab, MuJoCo Playground | PPO (adaptive learning rate from a KL target of 0.01, clip 0.2, GAE lambda 0.95), student-teacher distillation, random network distillation, and a symmetry extension (mirrored samples and a mirror loss, after Mittal et al., ICRA 2024) | Symmetry adopted as mirror images of skills (`mirrorSkill`, below). PPO not adopted: it needs thousands of parallel environments; the browser worker budget goes to evaluation and ES. |
| xbpeng/MimicKit | Apache-2.0 | Python; Isaac Gym, Isaac Lab or Newton | DeepMimic, AMP, AWR, ASE, LCP, ADD and SMP on reference motion clips (frames of root position, root rotation as an exponential map, and joint rotations) | Lesson only: skills as reference motions judged by goals, as MorphLab's keyframes already are. Adversarial motion priors need RL training and motion datasets (Tier C). |
| nicklashansen/tdmpc2 | MIT | GPU with at least 8 GB and 12 GB RAM for one task; 104 tasks; agents of 1M to 317M parameters | A learned latent world model planned with sampled action sequences (MPPI in the paper), trained by temporal-difference learning | Not adopted: training and planning at control rate exceed the browser. In MorphLab the simulator is the model, and ES searches skill parameters between runs. |
| danijar/dreamerv3 | MIT | JAX, Python 3.11+, GPU by default | A world model with categorical latents; an actor-critic trained on imagined trajectories; one fixed set of hyperparameters across domains | Not adopted (JAX training). Lesson kept: one robust configuration instead of per-task tuning, as MorphLab's single ES configuration for every taught skill. |
| real-stanford/diffusion_policy | MIT | PyTorch; Linux with an NVIDIA GPU for the benchmarks | Policies that denoise action sequences (observation, prediction and action horizons; executed receding-horizon); demonstrations in a replay buffer of arrays concatenated over episodes with episode ends | Not adopted: training is Tier C and iterative denoising does not fit a browser control loop. The dataset layout is noted for a possible demonstration export. |
| tonyzhaozh/act | MIT | PyTorch, MuJoCo 2.3.7, dm_control 1.0.14 | Action chunking with transformers (a CVAE; chunk size 100 in the examples) and temporal ensembling; about 90 % on transfer cube and 50 % on insertion in simulation | Not adopted: needs GPU training on teleoperated demonstrations; MorphLab's skills already play whole planned motions. |
| ARISE-Initiative/robomimic | MIT | PyTorch; robosuite datasets on HuggingFace | Learning from demonstration: behavior cloning variants (BC-RNN, BC-Transformer), offline RL (for example IQL), Diffusion Policy since v0.5; the CoRL 2021 study of what matters in offline demonstrations | Not adopted. MorphLab already has behavior cloning of a small MLP in the browser (the foot-lift imitation policy) and judges every policy by success over randomized trials, as the study recommends. |
| rail-berkeley/rlkit | MIT | PyTorch; MuJoCo 1.5 and gym 0.10.5; code last changed about five years ago | SAC, TD3, DQN, HER, AWAC, IQL, Skew-Fit | Not adopted (unmaintained, off-policy training with replay buffers). Lesson from hindsight relabeling: a run that missed its goal still shows what the parameters do; every try reports each measured goal for the next revision. |
| HybridRobotics/whole_body_tracking (BeyondMimic) | MIT | Isaac Lab 2.1.0, RSL-RL PPO; Unitree G1 | Humanoid motion tracking with DeepMimic-style rewards exp(-error / sigma^2), domain randomization, early termination, and adaptive sampling of motion start times in proportion to recent failures plus a uniform share of 0.1 | Lesson adopted in small form: when randomized trials expose failures (an evaluation below 80 %), the AI is told it can improve the skill over the same ranges and evaluate again. Failure-weighted sampling inside ES was not adopted; nothing measured here shows it would help skills of a few seconds. |

What changed in MorphLab (D26):
- `src/skills/mirror.ts`: the left-right mirror image of a taught skill.
  Limbs pair with the limb mounted at the mirror image of their mount;
  joints pair in order, and a value keeps its sign when the partner's axis
  is the mirror image of this one's and flips when it is the opposite (the
  hip roll of the quadruped preset, the coxa yaw of the hexapod, an arm's
  base yaw). Forward kinematics checks the map before it is used: the home
  stance must be its own mirror image, and every limb tip of a mirrored pose
  must sit at the mirror image of its partner's tip. Goals follow (left and
  right, turn direction, roll, limbs, the side of a reach), a gait's side
  step and turn change sign and each leg takes its partner's phase.
- The `mirrorSkill` tool and the Teach panel's Mirror left and right
  button make the other side's draft. It is tried and evaluated like any
  draft: a symmetric drawing does not prove symmetric physics.
- After an evaluation below 80 % of trials the AI is told it may improve
  the draft and evaluate again; the command then runs the version with the
  better evaluation.
