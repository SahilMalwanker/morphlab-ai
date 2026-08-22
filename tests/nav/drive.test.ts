import { describe, expect, it } from 'vitest';
import type { Vec3 } from '../../src/core/math/vec3';
import { PRESETS } from '../../src/core/model/catalog';
import { arm3, hexapod, quadruped, radialLegged, rover } from '../../src/core/model/presets';
import type { LimbDescription, RobotDescription } from '../../src/core/model/types';
import { validateRobot } from '../../src/core/model/validate';
import { driveGeometry, LEGGED_LIMITS, navConfigFor, placeAt, wheelTargets, WHEELED_LIMITS, type LeggedDrive, type WheeledDrive } from '../../src/nav/drive';
import { ROVER_HALF_EXTENTS } from '../../src/nav/referee';
import { DEFAULT_NAV, RAY_ORIGIN, ROVER_WHEELS } from '../../src/nav/stack';

function geometry(r: RobotDescription): WheeledDrive {
  const g = driveGeometry(r);
  if (!g.ok) throw new Error(g.reason);
  if (g.drive.kind !== 'wheeled') throw new Error(`${r.name} is not wheeled`);
  return g.drive;
}

function legged(r: RobotDescription): LeggedDrive {
  const g = driveGeometry(r);
  if (!g.ok) throw new Error(g.reason);
  if (g.drive.kind !== 'legged') throw new Error(`${r.name} is not legged`);
  return g.drive;
}

function reason(r: RobotDescription): string {
  const g = driveGeometry(r);
  if (g.ok) throw new Error(`${r.name} was accepted`);
  return g.reason;
}

/** The rover with every wheel joint changed by `edit`. */
function editedWheels(edit: (l: LimbDescription) => LimbDescription, name = 'Edited rover'): RobotDescription {
  const r = rover();
  return { ...r, id: 'edited_rover', name, limbs: r.limbs.map(edit) };
}

const withJoint = (l: LimbDescription, patch: { xyz?: Vec3; axis?: Vec3 }): LimbDescription => ({
  ...l,
  joints: l.joints.map((j) => ({ ...j, origin: { ...j.origin, xyz: patch.xyz ?? j.origin.xyz }, axis: patch.axis ?? j.axis })),
});

describe('drive geometry derived from the robot description', () => {
  it('reproduces, for the rover, the constants the navigation stack was tuned with', () => {
    const d = geometry(rover());
    expect(d.wheels).toEqual(ROVER_WHEELS);
    expect(d.wheelRadius).toBe(0.06);
    expect(d.trackWidth).toBe(0.34);
    expect(d.halfExtents).toEqual(ROVER_HALF_EXTENTS);
    expect(d.robotRadiusM).toBe(DEFAULT_NAV.robotRadiusM);
    expect(d.rayOrigin).toEqual(RAY_ORIGIN);
    expect(navConfigFor(d)).toEqual(DEFAULT_NAV);
    expect(d.limits).toEqual(WHEELED_LIMITS);
    expect(WHEELED_LIMITS).toEqual({ speedLimitRange: [0.1, 0.4], maxTurnRate: 1.8, remoteSpeed: 0.4, remoteTurn: 1.6, timeLimitS: 60 });
    // Wheel bottoms are 0.08 m below the base center; the rover keeps its own 0.1 m start height.
    expect(d.restHeight).toBe(0.08);
    expect(placeAt(rover(), d, { x: 0.5, y: -0.2, yaw: 0.3 }).base.pose).toEqual({ xyz: [0.5, -0.2, 0.1], rpy: [0, 0, 0.3] });
  });

  it('accepts the rover and the legged presets, and says why the arms cannot move', () => {
    const why = Object.fromEntries(PRESETS.map((p) => [p.id, driveGeometry(p.make())] as const));
    expect(why['rover']).toMatchObject({ ok: true, drive: { kind: 'wheeled' } });
    for (const id of ['quadruped', 'hexapod']) expect(why[id]).toMatchObject({ ok: true, drive: { kind: 'legged' } });
    for (const id of ['arm3', 'arm6', 'planar2']) expect(why[id]).toEqual({ ok: false, reason: 'its base is fixed in place, so it cannot move through the arena' });
  });

  it('derives track, footprint and planner clearance from an edited rover', () => {
    const wide = editedWheels((l) => withJoint(l, { xyz: [l.joints[0]!.origin.xyz[0], Math.sign(l.joints[0]!.origin.xyz[1]) * 0.25, -0.02] }), 'Wide rover');
    expect(validateRobot(wide)).toEqual([]);
    const d = geometry(wide);
    expect(d.trackWidth).toBe(0.5);
    // 0.25 m mount + half the 0.04 m wheel width + 0.02 m margin; the length is unchanged.
    expect(d.halfExtents).toEqual([0.22, 0.29]);
    // Half the footprint diagonal, sqrt(0.22^2 + 0.29^2) = 0.3640 m, plus 1.5 cm, rounded up to 1 cm.
    expect(d.robotRadiusM).toBe(0.38);
    const cfg = navConfigFor(d);
    expect([d.trackWidth, cfg.robotRadiusM, d.wheelRadius]).toEqual([0.5, 0.38, 0.06]);
    // The extra-cost band starts beyond the blocked band: 0.38 + 0.4 m.
    expect(cfg.softRangeM).toBeCloseTo(0.78, 12);
  });

  it('flips the command sign of wheels whose axis points to the right', () => {
    const flipped = editedWheels((l) => withJoint(l, { axis: [0, -1, 0] }));
    const d = geometry(flipped);
    expect(d.wheels.map((w) => w.sign)).toEqual([-1, -1, -1, -1]);
    expect(wheelTargets(d.wheels, 2, 3)).toEqual([-2, -3, -2, -3]);
    expect(wheelTargets(geometry(rover()).wheels, 2, 3)).toEqual([2, 3, 2, 3]);
  });

  it('refuses bodies the skid-steering controller cannot drive, with the reason', () => {
    const r = rover();
    expect(reason({ ...r, limbs: r.limbs.filter((l) => l.id === 'wheel_fl' || l.id === 'wheel_rl') })).toBe('skid steering needs at least one wheel on each side');
    expect(reason({ ...r, limbs: [...r.limbs, quadruped().limbs[0]!] })).toBe('it mixes wheels with legs or arms; Navigate drives bodies that roll on wheels only or walk on legs');
    expect(reason(editedWheels((l) => withJoint(l, { axis: [1, 0, 0] })))).toBe("the front left wheel does not spin about the body's left-right axis");
    expect(reason(editedWheels((l) => (l.id === 'wheel_rr' ? { ...l, wheel: { radius: 0.05, width: 0.04 } } : l)))).toBe('its wheels differ in radius; the skid-steering controller assumes one wheel radius');
    expect(reason(editedWheels((l) => ({ ...l, wheel: { radius: 0.01, width: 0.04 } })))).toBe('its body would rest on the floor instead of its wheels');
    expect(reason(editedWheels((l) => withJoint(l, { xyz: [l.joints[0]!.origin.xyz[0], l.joints[0]!.origin.xyz[1], -0.3] })))).toBe('its range rays would start 0.36 m above the floor and pass over the 0.3 m obstacles');
  });
});

describe('legged geometry derived from the robot description', () => {
  it('quadruped: stance height, footprint, ray mount and planner settings in closed form', () => {
    const d = legged(quadruped());
    // Thigh and shank (0.16 m each) at +-0.7 rad from vertical, plus the 0.022 m foot sphere.
    const rest = 2 * 0.16 * Math.cos(0.7) + 0.022;
    expect(d.restHeight).toBeCloseTo(rest, 5);
    expect(d.legs).toEqual([0, 1, 2, 3]);
    expect(d.stanceQ).toEqual(Array.from({ length: 4 }, () => [0, 0.7, -1.4]).flat());
    expect(d.gait).toMatchObject({ pattern: 'trot', cycleS: 0.5, strideM: 0, turnRad: 0, durationS: 20 });
    // Length: the rear knees, 0.16 sin 0.7 m behind the rear hips at -0.17 m, plus the 0.018 m link radius;
    // width: feet at 0.1 + 0.03 m plus the foot sphere. Both plus 4 cm for the steps and the 2 cm margin.
    expect(d.halfExtents[0]).toBeCloseTo(0.17 + 0.16 * Math.sin(0.7) + 0.018 + 0.06, 6);
    expect(d.halfExtents[1]).toBeCloseTo(0.13 + 0.022 + 0.06, 6);
    // sqrt(0.3511^2 + 0.212^2) = 0.4101 m, plus 1.5 cm, rounded up to 1 cm.
    expect(d.robotRadiusM).toBe(0.43);
    // Rays start 5 cm ahead of the body front (the front knees point back), 0.15 m above the floor.
    expect(d.rayOrigin[0]).toBeCloseTo(0.25, 6);
    expect(d.rayOrigin[1]).toBe(0);
    expect(d.rayOrigin[2]).toBeCloseTo(0.15 - rest, 5);
    expect(d.limits).toEqual(LEGGED_LIMITS);
    const cfg = navConfigFor(d);
    expect(cfg.follower).toEqual({ ...DEFAULT_NAV.follower, maxSpeed: 0.15, maxTurnRate: 0.5 });
    expect(cfg.rayHeightM).toBe(0.15);
    expect(cfg.softRangeM).toBeCloseTo(0.83, 12);
    // The description's 0.28 m start height is within 3 cm of standing.
    expect(placeAt(quadruped(), d, { x: 0, y: 0, yaw: 0 }).base.pose.xyz[2]).toBe(0.28);
  });

  it('hexapod: legs spread sideways make the footprint wider than long', () => {
    const d = legged(hexapod());
    expect(d.legs).toHaveLength(6);
    // Femur 0.1 m at 0.3 rad and tibia 0.12 m at 1.5 rad below horizontal, plus the 0.02 m foot sphere.
    const rest = 0.1 * Math.sin(0.3) + 0.12 * Math.sin(1.5) + 0.02;
    expect(d.restHeight).toBeCloseTo(rest, 5);
    const reach = 0.12 + 0.05 + 0.1 * Math.cos(0.3) + 0.12 * Math.cos(1.5) + 0.02;
    expect(d.halfExtents[0]).toBeCloseTo(0.18 + 0.06, 6);
    expect(d.halfExtents[1]).toBeCloseTo(reach + 0.06, 5);
    expect(d.robotRadiusM).toBe(0.45);
    expect(d.rayOrigin[0]).toBeCloseTo(0.23, 6);
    expect(d.restHeight + d.rayOrigin[2]).toBeCloseTo(0.15, 6);
  });

  it('refuses bodies that cannot walk with the gait generator, with the reason', () => {
    expect(reason(radialLegged(2, 3))).toBe('it has 2 legs; walking in Navigate needs at least four, so the body stays up while half of them step');
    expect(reason(radialLegged(4, 1))).toBe('the leg 1 has one joint; each leg needs at least two to step');
    const arm = arm3();
    expect(reason({ ...arm, base: { ...arm.base, kind: 'floating' } })).toBe('it has neither wheels nor legs to move with');
    const dog = quadruped();
    expect(reason({ ...dog, stance: { ...dog.stance, front_left_hip_pitch: 1.6 } })).toMatch(/^in its stance its feet are not level \(they differ by 0\.\d\d m\), so not every foot reaches the floor$/);
  });
});
