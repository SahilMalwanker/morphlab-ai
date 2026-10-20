import * as z from 'zod';
import type { RobotCapabilities } from './capabilities.ts';
import { normalizePlaceName, PLACE_NAME_PATTERN } from './memory.ts';
import { rejected, type ToolResult } from './result.ts';

const ident = z.string().min(1).max(64).regex(/^[A-Za-z0-9_.:-]+$/);
/** Skill names a model may use; library names look like "Lift front left (state machine)". */
export const SKILL_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9 _.,()'+-]{0,79}$/;
const skillName = z.string().min(1).max(80).regex(SKILL_NAME_PATTERN);
const placeName = z.string().min(1).max(40).regex(PLACE_NAME_PATTERN);
const maxSpeed = z.number().min(0.1).max(0.4).optional().describe('Speed limit in m/s (wheeled robots 0.1 to 0.4, legged robots up to 0.15); omit to keep the current limit');
const MAX_ARGUMENT_CHARS = 4000;

// src/ai is shared with the proxy and imports nothing from src/skills; tests check these copies against the originals.
/** A skill draft's name: TAUGHT_NAME_PATTERN in src/skills/taught.ts. */
export const DRAFT_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9 _'-]{0,39}$/;
/** Ranges of goal values: GOAL_BOUNDS in src/skills/taught.ts. */
export const DRAFT_GOAL_BOUNDS = {
  maxTiltRad: [0.05, 1.6],
  minBaseHeightM: [0, 2],
  minHeightM: [0, 2],
  holdS: [0, 10],
  heightM: [0, 2],
  minS: [0.01, 2],
  minRiseM: [0.005, 1],
  minDistanceM: [0.01, 5],
  maxDistanceM: [0.01, 5],
  minAngleRad: [0.05, 6.3],
  maxAngleRad: [0.05, 6.3],
  minRad: [-1.6, 1.6],
  maxRad: [-1.6, 1.6],
  minFraction: [0, 1],
  minAmplitudeM: [0.005, 1],
  minSwings: [1, 20],
  x: [-2, 2],
  y: [-2, 2],
  z: [-2, 2],
  toleranceM: [0.005, 0.5],
} as const;
/** Ranges of composed poses: COMPOSE_LIMITS and MAX_COMPOSED_KEYFRAMES in src/skills/compose.ts. */
export const DRAFT_POSE_LIMITS = { bodyShiftM: 0.2, bodyAngleRad: 0.8, tipOffsetM: 0.5, secondsMin: 0.1, secondsMax: 10 } as const;
export const MAX_DRAFT_KEYFRAMES = 8;
/** Gait patterns and ranges: GAIT_PATTERNS and GAIT_BOUNDS in src/skills/gait.ts. */
export const DRAFT_GAIT_PATTERNS = ['trot', 'walk', 'pace', 'bound', 'pronk', 'wave', 'stand'] as const;
export const DRAFT_GAIT_BOUNDS = {
  dutyFactor: [0.2, 1],
  cycleS: [0.25, 3],
  strideM: [-0.3, 0.3],
  lateralM: [-0.2, 0.2],
  turnRad: [-0.8, 0.8],
  stepHeightM: [0, 0.15],
  bodyHeightM: [-0.12, 0.06],
  durationS: [0.5, 20],
  feedbackKp: [0, 0.5],
  feedbackKd: [0, 0.05],
} as const;

const goalValue = (key: keyof typeof DRAFT_GOAL_BOUNDS) => z.number().min(DRAFT_GOAL_BOUNDS[key][0]).max(DRAFT_GOAL_BOUNDS[key][1]);
const gaitValue = (key: keyof typeof DRAFT_GAIT_BOUNDS) => z.number().min(DRAFT_GAIT_BOUNDS[key][0]).max(DRAFT_GAIT_BOUNDS[key][1]);
const bodyShift = z.number().min(-DRAFT_POSE_LIMITS.bodyShiftM).max(DRAFT_POSE_LIMITS.bodyShiftM).optional();
const bodyAngle = z.number().min(-DRAFT_POSE_LIMITS.bodyAngleRad).max(DRAFT_POSE_LIMITS.bodyAngleRad).optional();
const tipOffset = z.number().min(-DRAFT_POSE_LIMITS.tipOffsetM).max(DRAFT_POSE_LIMITS.tipOffsetM).optional();

const draftPose = z.strictObject({
  seconds: z.number().min(DRAFT_POSE_LIMITS.secondsMin).max(DRAFT_POSE_LIMITS.secondsMax).describe('Time to reach this pose from the previous one (the first from the home stance), s'),
  body: z
    .strictObject({ x: bodyShift, y: bodyShift, z: bodyShift, roll: bodyAngle, pitch: bodyAngle, yaw: bodyAngle })
    .optional()
    .describe('Floating bases only: move the body from its home stance (m, rad; pitch above 0 lowers the front, roll above 0 lowers the right side) while the feet stay where they stand'),
  tips: z
    .array(z.strictObject({ limb: ident, x: tipOffset, y: tipOffset, z: tipOffset }))
    .max(12)
    .optional()
    .describe('Move limb tips from where they rest in this pose, m, in the home body frame (x forward, y left, z up); z above 0 lifts a foot'),
  joints: z
    .array(z.strictObject({ joint: ident, value: z.number().min(-6.3).max(6.3) }))
    .max(48)
    .optional()
    .describe("Joint targets in rad (m for sliding joints) within each joint's limits, set after the tips"),
}).describe('A whole pose from the home stance: a body move or tip not given is back at home, so repeat it in each pose that keeps it; repeating a pose holds it');

const draftGoal = z.discriminatedUnion('kind', [
  z.strictObject({ kind: z.literal('upright'), maxTiltRad: goalValue('maxTiltRad') }),
  z.strictObject({ kind: z.literal('noFall'), minBaseHeightM: goalValue('minBaseHeightM') }),
  z.strictObject({ kind: z.literal('tipHeight'), limbId: ident, minHeightM: goalValue('minHeightM'), holdS: goalValue('holdS') }),
  z.strictObject({ kind: z.literal('baseHeight'), below: z.boolean(), heightM: goalValue('heightM'), holdS: goalValue('holdS') }),
  z.strictObject({ kind: z.literal('airborne'), minS: goalValue('minS'), limbIds: z.array(ident).min(1).max(8).optional() }),
  z.strictObject({ kind: z.literal('baseRise'), minRiseM: goalValue('minRiseM') }),
  z.strictObject({ kind: z.literal('travel'), direction: z.enum(['forward', 'backward', 'left', 'right']), minDistanceM: goalValue('minDistanceM'), maxDistanceM: goalValue('maxDistanceM').optional() }),
  z.strictObject({ kind: z.literal('turn'), left: z.boolean(), minAngleRad: goalValue('minAngleRad'), maxAngleRad: goalValue('maxAngleRad').optional() }),
  z.strictObject({ kind: z.literal('standsAtEnd'), maxTiltRad: goalValue('maxTiltRad'), minBaseHeightM: goalValue('minBaseHeightM') }),
  z.strictObject({ kind: z.literal('bodyAngle'), axis: z.enum(['pitch', 'roll']), minRad: goalValue('minRad'), maxRad: goalValue('maxRad'), holdS: goalValue('holdS') }),
  z.strictObject({ kind: z.literal('footDown'), limbId: ident, minFraction: goalValue('minFraction') }),
  z.strictObject({ kind: z.literal('tipSwing'), limbId: ident, axis: z.enum(['x', 'y', 'z']), minAmplitudeM: goalValue('minAmplitudeM'), minSwings: z.int().min(1).max(20) }),
  z.strictObject({ kind: z.literal('tipReach'), limbId: ident, x: goalValue('x'), y: goalValue('y'), z: goalValue('z'), toleranceM: goalValue('toleranceM'), holdS: goalValue('holdS') }),
]);

const draftGait = z.strictObject({
  pattern: z.enum(DRAFT_GAIT_PATTERNS).describe('trot: diagonal legs together (a tripod on six legs), the default for walking; walk: one leg at a time, rear to front, needs dutyFactor of at least 1 - 1/legs; pace: one side together; bound: front then rear; pronk: all together; wave: one leg at a time in turn, same dutyFactor rule as walk; stand: no steps'),
  phases: z.array(z.number().min(0).max(1)).max(12).optional().describe('Custom contact pattern: phase offset per limb in limb order, fractions of a cycle; replaces the pattern'),
  heldUp: z.array(ident).max(4).optional().describe('Legs held off the ground the whole time (a three-legged walk)'),
  dutyFactor: gaitValue('dutyFactor').describe('Fraction of each cycle a foot is on the ground'),
  cycleS: gaitValue('cycleS').describe('Duration of one step cycle, s'),
  strideM: gaitValue('strideM').describe('Body travel per cycle along its x axis, m (negative walks backward)'),
  lateralM: gaitValue('lateralM').default(0).describe('Body travel per cycle along its y axis, m (positive steps left)'),
  turnRad: gaitValue('turnRad').default(0).describe('Heading change per cycle, rad (positive turns left)'),
  stepHeightM: gaitValue('stepHeightM').describe('Foot lift during a step, m'),
  bodyHeightM: gaitValue('bodyHeightM').default(0).describe('Body height change from the home stance, m'),
  durationS: gaitValue('durationS').describe('How long to step, s'),
  feedbackKp: gaitValue('feedbackKp').default(0.1).describe('IMU feedback: foot height change per rad of roll or pitch, m/rad; 0 is open loop'),
  feedbackKd: gaitValue('feedbackKd').default(0.005).describe('Foot height change per rad/s of roll or pitch rate'),
});

const GOALS_TEXT =
  'What success means, judged from simulator ground truth: upright (tilt at most maxTiltRad), noFall (body above minBaseHeightM), tipHeight (that limb tip at least minHeightM above the ground for holdS), baseHeight (body below or above heightM for holdS), airborne (every foot off the ground together with the body upright for minS, or with limbIds only those feet, the others may stay down: standing on two legs names the two lifted feet; a jump names none and also needs standsAtEnd or upright, so it lands on its feet), baseRise (body rises minRiseM above its resting height), travel (body ends minDistanceM forward, backward, left or right of its start, and at most maxDistanceM when given), turn (body turns minAngleRad left or right, at most maxAngleRad when given; give both bounds for "about" an amount), standsAtEnd (tilt at most maxTiltRad and body above minBaseHeightM over the last 0.5 s), bodyAngle (body pitch or roll between minRad and maxRad for holdS; positive pitch lowers the front, positive roll raises the left side), footDown (that foot on the ground for at least minFraction of the run), tipSwing (that limb tip swings along x, y or z at least minSwings times, each swing one movement one way of at least minAmplitudeM, so back and forth is two), tipReach (that limb tip within toleranceM of a point x, y, z m from where it started, x forward, y left, z up, for holdS). A fixed base takes tipHeight, tipSwing and tipReach goals only.';

export interface ToolDefinition {
  name: string;
  description: string;
  schema: z.ZodType;
  /** Null when the tool is available, otherwise the reason, phrased for the user. */
  unavailable: (caps: RobotCapabilities) => string | null;
  /** Checks across arguments that JSON Schema cannot express; null when they pass. */
  check?: (args: Record<string, unknown>) => string | null;
}

const always = (): null => null;

export const TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  {
    name: 'getRobotCapabilities',
    description: 'List what the current robot can do: body type, joints, saved skills and supported actions.',
    schema: z.strictObject({}),
    unavailable: always,
  },
  {
    name: 'getRobotState',
    description: 'Summarize the measured robot state: pose or joint positions, range-sensor distances where the robot has them, the active task and any fault.',
    schema: z.strictObject({}),
    unavailable: always,
  },
  {
    name: 'getSceneObjects',
    description: 'List scene objects with id, kind, color and position in m, labeled by source (sensor or debug ground truth).',
    schema: z.strictObject({
      mode: z.enum(['sensor', 'debug']).default('sensor').describe('sensor: only what the camera and sensors report; debug: simulator ground truth'),
    }),
    unavailable: (c) => (c.features.sceneObjects ? null : 'this scene has no objects to report.'),
  },
  {
    name: 'pointAt',
    description: 'Point the end effector at a scene object given by id; the tool layer looks up its position.',
    schema: z.strictObject({
      targetId: ident,
      standoff: z.number().min(0.05).max(0.5).default(0.15).describe('Distance between end effector and target in m'),
    }),
    unavailable: (c) =>
      !c.features.pointing ? 'pointing needs an arm with at least three joints.' : c.features.sceneObjects ? null : 'there are no scene objects to point at.',
  },
  {
    name: 'navigateTo',
    description: 'Drive or walk the Navigate robot to a goal on the arena floor (world frame, m) with its navigation stack, and wait for the measured outcome.',
    schema: z.strictObject({
      x: z.number().min(-0.4).max(3.4).describe('Goal x in m (arena: -0.4 to 3.4)'),
      y: z.number().min(-1.9).max(1.9).describe('Goal y in m (arena: -1.9 to 1.9)'),
      maxSpeed,
    }),
    unavailable: (c) => (c.features.navigation ? null : 'this robot cannot drive.'),
  },
  {
    name: 'navigateToPlace',
    description: 'Drive or walk the Navigate robot to a remembered place on this map, by name, and wait for the measured outcome.',
    schema: z.strictObject({ place: placeName.describe('Name of a remembered place'), maxSpeed }),
    unavailable: (c) => (!c.features.navigation ? 'this robot cannot drive.' : c.places.length > 0 ? null : 'no places are remembered on this map.'),
  },
  {
    name: 'rememberPlace',
    description: "Remember a named place on this map. Without x and y the tool reads the robot's current position itself (the controller's estimate in sensor-only mode); with x and y it stores those. Use it when the user asks to remember, save or mark a spot. A place with the same name is replaced.",
    schema: z.strictObject({
      name: placeName.describe('Short name, for example "dock"'),
      x: z.number().min(-0.4).max(3.4).optional().describe('x in m; give x and y together, or neither for the current position'),
      y: z.number().min(-1.9).max(1.9).optional().describe('y in m'),
    }),
    unavailable: (c) => (c.features.navigation ? null : 'only the rover remembers places.'),
    check: (a) => ((a['x'] === undefined) === (a['y'] === undefined) ? null : 'give x and y together, or neither.'),
  },
  {
    name: 'forgetPlace',
    description: 'Forget a remembered place on this map.',
    schema: z.strictObject({ place: placeName }),
    unavailable: (c) => (c.features.navigation && c.places.length > 0 ? null : 'no places are remembered on this map.'),
  },
  {
    name: 'executeSkill',
    description: 'Run a saved, evaluated skill by its library name (a skill the user taught, or a foot lift) and wait for the measured outcome.',
    schema: z.strictObject({
      skill: skillName,
      holdSeconds: z.number().min(0).max(10).optional().describe('Foot lifts only: how long to hold the foot up, s. Omit it for skills the user taught; they keep their own timing.'),
    }),
    unavailable: (c) => (c.skills.length > 0 ? null : 'no evaluated skills are saved for this robot.'),
  },
  {
    name: 'draftSkill',
    description:
      'Compose a new skill for a motion that no saved skill does: either poses (keyframes reached one after another, the last one held) or a gait (stepping from a contact pattern, for walking, turning or stepping in place), and goals that measure what the user asked for. The draft goes to the Teach panel, where the user sees and can edit it, and is tried once in the live simulation; the result gives each goal as measured. Nothing is saved; evaluateSkillDraft evaluates and saves it.',
    schema: z.strictObject({
      name: z.string().min(1).max(40).regex(DRAFT_NAME_PATTERN).describe('Short name that commands will use, for example "wave" or "sit"'),
      keyframes: z.array(draftPose).min(1).max(MAX_DRAFT_KEYFRAMES).optional().describe('Poses relative to the home stance; anything a pose leaves out stays at home'),
      gait: draftGait.optional().describe('Legged robots: step from a contact pattern instead of playing poses'),
      finalHoldSeconds: z.number().min(0).max(10).default(1).describe('How long to hold the last pose, s (keyframes only)'),
      goals: z.array(draftGoal).min(1).max(6).describe(GOALS_TEXT),
    }),
    unavailable: (c) => (c.features.skillAuthoring ? null : 'skills are composed in the Skills tab.'),
    check: (a) => ((a['keyframes'] === undefined) === (a['gait'] === undefined) ? 'give either keyframes or a gait.' : null),
  },
  {
    name: 'improveSkillDraft',
    description:
      'Improve the skill draft in the Teach panel by trial and error: evolution strategies adjust its poses and timing (or its gait parameters and feedback gains) to raise the goal reward in randomized simulated trials, and keep the change only if it scores better on held-out trials. Takes up to a minute; call evaluateSkillDraft afterwards.',
    schema: z.strictObject({}),
    unavailable: (c) => (c.features.skillAuthoring ? null : 'skill drafts are improved in the Skills tab.'),
  },
  {
    name: 'trainGaitPolicy',
    description:
      "Train a learned feedback policy for the skill draft in the Teach panel by augmented random search. Gait draft: a small network adds corrections (up to 0.15 rad) to every joint target every 10 ms from the IMU and the gait phase, rewarded for keeping the gait's own speed, no drift, staying upright and meeting the goals (up to two minutes). Keyframe draft on a free-standing body: reinforcement learning in randomized runs in which the body is pushed; the network adds up to 0.5 rad to every joint target every 10 ms from the IMU, the foot contacts and the joint encoders, rewarded by the goals holding at each moment and at the end (about five minutes). This makes an unstable pose (two legs, the hind legs) or a landing closed loop. The draft keeps the policy only if it beats the draft alone on held-out runs (pushed, for keyframes) without more falls. Call evaluateSkillDraft afterwards.",
    schema: z.strictObject({ architecture: z.enum(['linear', 'mlp']).optional().describe('Default linear. For keyframe drafts only, mlp trains an experimental 16-hidden-unit nonlinear policy locally in CPU workers. It must beat an existing draft policy too before replacing it. No GPU or remote training is used.') }),
    unavailable: (c) => (!c.features.skillAuthoring ? 'feedback policies are trained in the Skills tab.' : c.bodyKind === 'legged' ? null : 'only free-standing legged robots get a learned feedback policy.'),
  },
  {
    name: 'trainLocomotion',
    description:
      "Train this legged body's walking controller, once per body (four to eight minutes on four background workers): the gait generator is measured under a set of commands and a calibration from speed asked for to generator command is fitted; a feedback policy that reads the IMU and a sensor-only estimate of where the body is (leg odometry) against where the command leads, and changes the generator's stride, side step and turn every 10 ms, is trained by augmented random search over random speed commands under randomized friction, mass and joint offsets, and kept only if it beats the calibrated generator on held-out runs; the result is evaluated over 40 randomized commands against the generator without calibration. Afterwards locomote walks by speed or to a target, and Navigate walks paths with it.",
    schema: z.strictObject({}),
    unavailable: (c) => (c.bodyKind === 'legged' && (c.features.skillAuthoring || c.features.navigation) ? null : !c.features.skillAuthoring && !c.features.navigation ? 'walking controllers are trained in the Skills tab.' : 'only legged robots walk with gaits.'),
  },
  {
    name: 'locomote',
    description:
      "Walk with this body's learned walking controller and report where the body went (simulator ground truth) against where the command leads. Give either a target, where the body should end up measured from where it stands now in its own frame (forwardM, leftM, turnRad; the tool picks speeds within the controller's ranges and the duration, and walking and turning together follow an arc), or speeds in the body frame with durationS (stepping fades in and out over 0.5 s, so distance is about speed x (durationS - 0.5)). Walk again to correct a remaining error.",
    schema: z.strictObject({
      forwardM: z.number().min(-2).max(2).optional().describe('Target: how far forward of here to end up, m (negative behind)'),
      leftM: z.number().min(-1).max(1).optional().describe('Target: how far to the left of here to end up, m (negative right)'),
      turnRad: z.number().min(-3.2).max(3.2).optional().describe('Target: heading change, rad (positive turns left)'),
      forwardMps: z.number().min(-0.5).max(0.5).optional().describe('Instead of a target: forward speed, m/s (negative walks backward)'),
      leftMps: z.number().min(-0.3).max(0.3).optional().describe('Instead of a target: sideways speed, m/s (positive steps left, negative right)'),
      turnRadps: z.number().min(-1.5).max(1.5).optional().describe('Instead of a target: turn rate, rad/s (positive turns left)'),
      durationS: z.number().min(1).max(10).optional().describe('With speeds: how long to walk, s'),
    }),
    unavailable: (c) => (!c.features.skillAuthoring || c.bodyKind !== 'legged' ? 'walking by command runs in the Skills tab on a legged robot.' : c.walking ? null : 'no walking controller is trained for this body yet; call trainLocomotion first.'),
    check: (a) => {
      const target = ['forwardM', 'leftM', 'turnRad'].some((k) => a[k] !== undefined);
      const speeds = ['forwardMps', 'leftMps', 'turnRadps', 'durationS'].some((k) => a[k] !== undefined);
      if (target === speeds) return 'give either a target (forwardM, leftM, turnRad) or speeds with durationS.';
      return speeds && a['durationS'] === undefined ? 'durationS is required with speeds.' : null;
    },
  },
  {
    name: 'mirrorSkill',
    description:
      'Make the left-right mirror image of a saved skill for the other side, instead of composing it again: limbs swap sides, sideways steps, turns and roll change sign, and the goals follow (a front left wave from a front right one, a right turn from a left turn). The body is checked for symmetry first. The mirrored draft goes to the Teach panel and is tried once like draftSkill; evaluateSkillDraft evaluates and saves it.',
    schema: z.strictObject({
      skill: skillName.describe('The saved skill to mirror'),
      name: z.string().min(1).max(40).regex(DRAFT_NAME_PATTERN).optional().describe('Name of the mirrored skill; omit it to swap left and right in the name'),
    }),
    unavailable: (c) => (!c.features.skillAuthoring ? 'skills are mirrored in the Skills tab.' : c.skills.length > 0 ? null : 'no evaluated skills are saved for this robot.'),
  },
  {
    name: 'evaluateSkillDraft',
    description:
      'Evaluate the skill draft in the Teach panel over 20 randomized trials (friction, body mass and joint offsets vary) and save it to the library when at least one trial meets every goal. Returns the success rate per goal and the saved name, which executeSkill runs.',
    schema: z.strictObject({}),
    unavailable: (c) => (c.features.skillAuthoring ? null : 'skill drafts are evaluated in the Skills tab.'),
  },
  {
    name: 'runSkillSequence',
    description:
      'Run saved skills one after another, each from the home stance and judged on its own, stopping at the first run that does not succeed; reports every run as measured and the chance of success estimated beforehand from the evaluations. Use it for requests such as "hop twice" or "sit, then wave".',
    schema: z.strictObject({
      steps: z.array(z.strictObject({ skill: skillName, times: z.int().min(1).max(5).default(1) })).min(1).max(8),
    }),
    unavailable: (c) => (c.features.skillSequences && c.skills.length > 0 ? null : 'no evaluated skills are saved for this robot.'),
  },
  {
    name: 'selectRobot',
    description: "Switch this tab's robot. Skills: a preset (quadruped, hexapod, arm3, arm6) or the Build tab's robot ('built'); saved skills change with it, as they are bound to their robot. Navigate: rover, quadruped, hexapod or 'built'.",
    schema: z.strictObject({ robot: z.enum(['rover', 'quadruped', 'hexapod', 'arm3', 'arm6', 'built']) }),
    unavailable: (c) => (c.features.robotChoice ? null : 'robots are switched in the Skills tab.'),
  },
  {
    name: 'pushRobot',
    description: 'Push the robot in the Physics tab with a short impulse and report how it responded over the next 2 s of simulated time (largest tilt, lowest body height, whether it stayed up), measured in the simulator.',
    schema: z.strictObject({
      direction: z.enum(['forward', 'backward', 'left', 'right']),
      impulseNs: z.number().min(0.2).max(5).describe('Impulse in N*s, applied over 50 ms'),
    }),
    unavailable: (c) => (c.features.physicsTools ? null : 'pushes are applied in the Physics tab.'),
  },
  {
    name: 'resetSimulation',
    description: 'Reset the Physics tab simulation: the robot returns to its start state and home pose.',
    schema: z.strictObject({}),
    unavailable: (c) => (c.features.physicsTools ? null : 'this view has no simulation to reset.'),
  },
  {
    name: 'moveTipTo',
    description: 'Kinematics tab (geometry only, no physics): solve inverse kinematics so a limb tip reaches a world point, m, and report the solver status, the remaining distance and the joint values.',
    schema: z.strictObject({
      limb: ident.optional().describe('Limb id; omit for the selected limb'),
      x: z.number().min(-3).max(3),
      y: z.number().min(-3).max(3),
      z: z.number().min(-3).max(3),
    }),
    unavailable: (c) => (!c.features.kinematicsTools ? 'inverse kinematics runs in the Kinematics tab.' : c.targetableLimbs?.length === 0 ? 'this robot has no limb that can follow a position target.' : null),
  },
  {
    name: 'setJointAngles',
    description: 'Kinematics tab (geometry only, no physics): set joint angles in rad (m for sliding joints); values are clamped to the joint limits, and the resulting tip positions are reported.',
    schema: z.strictObject({ joints: z.array(z.strictObject({ joint: ident, value: z.number().min(-6.3).max(6.3) })).min(1).max(48) }),
    unavailable: (c) => (c.features.kinematicsTools ? null : 'joint angles are set in the Kinematics tab.'),
  },
  {
    name: 'buildRobot',
    description: 'Build tab: design a robot from a preset, as a legged robot (leg pairs, joints per leg, body and leg sizes, joint effort) or as an arm (joints, reach). Every design is validated; a refused one names the problem and changes nothing.',
    schema: z.strictObject({
      preset: z.enum(['quadruped', 'hexapod', 'arm3', 'arm6', 'planar2', 'rover']).optional(),
      legged: z
        .strictObject({
          pairs: z.int().min(1).max(4),
          dofPerLeg: z.int().min(1).max(4),
          bodyLength: z.number().min(0.1).max(1.5),
          bodyWidth: z.number().min(0.08).max(1),
          bodyHeight: z.number().min(0.03).max(0.5),
          bodyMass: z.number().min(0.5).max(40),
          thigh: z.number().min(0.05).max(0.6),
          shank: z.number().min(0.05).max(0.6),
          effort: z.number().min(1).max(200).describe('Joint effort limit, N*m'),
        })
        .optional(),
      arm: z.strictObject({ dof: z.int().min(1).max(7), reach: z.number().min(0.2).max(2).describe('Total link length, m') }).optional(),
      name: z.string().min(1).max(40).regex(DRAFT_NAME_PATTERN).optional(),
    }),
    unavailable: (c) => (c.features.designTools ? null : 'robots are designed in the Build tab.'),
    check: (a) => ([a['preset'], a['legged'], a['arm']].filter((v) => v !== undefined).length === 1 ? null : 'give exactly one of preset, legged or arm.'),
  },
  {
    name: 'modifyTask',
    description: 'Adjust the running task: slower, faster, higher, lower, closer or farther, by a little or a normal step.',
    schema: z.strictObject({
      taskId: ident,
      change: z.enum(['slower', 'faster', 'higher', 'lower', 'closer', 'farther']),
      amount: z.enum(['little', 'normal']).default('normal'),
    }),
    unavailable: always,
  },
  {
    name: 'cancelTask',
    description: 'Cancel the running task; the robot holds, brakes or stands depending on its body.',
    schema: z.strictObject({ taskId: ident }),
    unavailable: always,
  },
  {
    name: 'listSkills',
    description: 'List saved skills for this robot with their evaluated success; with a skill name, that skill\'s method, goals and live run record.',
    schema: z.strictObject({ skill: skillName.optional().describe('A saved skill name for its details; omit for the list') }),
    unavailable: always,
  },
  {
    name: 'saveSkillFromDemonstration',
    description: 'Turn a recorded demonstration into a named skill.',
    schema: z.strictObject({ name: skillName, demonstrationId: ident }),
    unavailable: (c) => (c.features.demonstrations ? null : 'recording demonstrations is not available yet.'),
  },
  {
    name: 'evaluateSkill',
    description: 'Run randomized trials of a saved skill and report the measured success rate.',
    schema: z.strictObject({ skill: skillName, trials: z.int().min(5).max(50).default(20) }),
    unavailable: (c) =>
      !c.features.skillEvaluation ? 'skill evaluation is not available for this robot.' : c.skills.length > 0 ? null : 'no skills are saved for this robot.',
  },
  {
    name: 'runControllerExperiment',
    description: "Run the Control lab's step test and report the measured step metrics. Without a configuration it runs the lab as set (gain set A, and gain set B when comparison is on) and changes nothing. With a configuration it first sets gain set B to it, then runs gain sets A and B. Gain set A is the learner's and is never changed.",
    schema: z.strictObject({
      configuration: z
        .strictObject({
          kp: z.number().min(1).max(400).describe('Proportional gain in N*m/rad (N/m for sliding joints)'),
          kd: z.number().min(0).max(40).describe('Derivative gain in N*m*s/rad (N*s/m for sliding joints)'),
          gravityCompensation: z.boolean().default(false),
        })
        .optional()
        .describe('Gains to try on gain set B; omit to run the current setup unchanged'),
    }),
    unavailable: (c) => (c.features.controllerExperiments ? null : 'controller experiments are not available for this robot.'),
  },
  {
    name: 'saveSkill',
    description: 'Save an evaluated artifact from evaluateSkill or runControllerExperiment as a named skill.',
    schema: z.strictObject({ name: skillName, artifactId: ident }),
    unavailable: (c) => (c.features.skillEvaluation ? null : 'skills cannot be saved for this robot.'),
  },
];

export const TOOL_NAMES: ReadonlySet<string> = new Set(TOOL_DEFINITIONS.map((t) => t.name));

export interface ModelToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

/** Tool specifications for a model, limited to what the current robot can do. */
export function toolsForModel(caps: RobotCapabilities): ModelToolSpec[] {
  return TOOL_DEFINITIONS.filter((t) => t.unavailable(caps) === null).map((t) => {
    const schema = z.toJSONSchema(t.schema, { io: 'input' });
    const parameters: Record<string, unknown> = Object.fromEntries(Object.entries(schema).filter(([key]) => key !== '$schema'));
    return { name: t.name, description: t.description, parameters };
  });
}

export type ValidatedCall = { ok: true; name: string; args: Record<string, unknown> } | { ok: false; result: ToolResult };

function safeName(name: string): string {
  return name.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64) || 'that name';
}

function formatIssues(issues: ReadonlyArray<{ path: ReadonlyArray<PropertyKey>; message: string }>): string {
  return issues
    .slice(0, 3)
    .map((i) => `${i.path.map(String).join('.') || 'arguments'}: ${i.message}`)
    .join('; ');
}

/** Validates a model's tool call before anything runs: known tool, available on this robot, bounded arguments. */
export function validateToolCall(name: string, rawArgs: unknown, caps: RobotCapabilities): ValidatedCall {
  const tool = TOOL_DEFINITIONS.find((t) => t.name === name);
  if (!tool) return { ok: false, result: rejected(`There is no tool named ${safeName(name)}.`) };
  const why = tool.unavailable(caps);
  if (why) return { ok: false, result: rejected(`Cannot do this yet: ${why}`) };
  let input: unknown = rawArgs ?? {};
  if (typeof rawArgs === 'string') {
    if (rawArgs.length > MAX_ARGUMENT_CHARS) return { ok: false, result: rejected('The tool arguments are too long.') };
    try {
      input = rawArgs.trim() === '' ? {} : JSON.parse(rawArgs);
    } catch {
      return { ok: false, result: rejected('The tool arguments are not valid JSON.') };
    }
  }
  const parsed = tool.schema.safeParse(input);
  if (!parsed.success) return { ok: false, result: rejected(`Invalid arguments for ${tool.name}: ${formatIssues(parsed.error.issues)}`) };
  const args = parsed.data as Record<string, unknown>;
  const problem = tool.check?.(args);
  if (problem) return { ok: false, result: rejected(`Invalid arguments for ${tool.name}: ${problem}`) };
  // A saved skill may be named without its method suffix ("hop" for "hop (keyframes)") when only one matches.
  const resolve = (name: unknown): unknown => {
    if (typeof name !== 'string' || caps.skills.includes(name)) return name;
    const key = name.trim().toLowerCase();
    const hits = caps.skills.filter((s) => s.toLowerCase() === key || s.toLowerCase().startsWith(`${key} (`));
    return hits.length === 1 ? hits[0] : name;
  };
  if (args['skill'] !== undefined) args['skill'] = resolve(args['skill']);
  if (Array.isArray(args['steps'])) args['steps'] = (args['steps'] as Record<string, unknown>[]).map((s) => ({ ...s, skill: resolve(s['skill']) }));
  const skill = args['skill'];
  const steps = Array.isArray(args['steps']) ? (args['steps'] as { skill?: unknown }[]).map((s) => s.skill) : [];
  const unsaved = [skill, ...steps].find((s) => typeof s === 'string' && !caps.skills.includes(s));
  if (typeof unsaved === 'string') {
    const fallback = caps.skills.length > 0 ? `Saved skills: ${caps.skills.join(', ')}.` : 'Teach and evaluate a skill first.';
    return { ok: false, result: rejected(`No evaluated skill named "${unsaved}" is saved for ${caps.name}.`, fallback) };
  }
  const place = args['place'];
  if (typeof place === 'string') {
    const key = normalizePlaceName(place).toLowerCase();
    if (!caps.places.some((p) => normalizePlaceName(p).toLowerCase() === key)) {
      const fallback = caps.places.length > 0 ? `Remembered places: ${caps.places.join(', ')}.` : 'Remember a place first.';
      return { ok: false, result: rejected(`No place named "${normalizePlaceName(place)}" is remembered on this map.`, fallback) };
    }
  }
  return { ok: true, name: tool.name, args };
}
