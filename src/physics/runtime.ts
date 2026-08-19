import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';
import { fromRotationTranslation, rpyToQuat, type Mat4, type Quat } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import type { Pose, RobotDescription } from '../core/model/types';
import { generateMjcf, mjcfNames, type MjcfOptions } from './mjcf';

export interface RuntimeLayout {
  nq: number;
  nv: number;
  nu: number;
  floating: boolean;
  rootQposAdr: number;
  rootDofAdr: number;
  baseBodyId: number;
  jointIds: string[];
  qposAdr: number[];
  dofAdr: number[];
  actuatorIndex: number[];
  linkBodyIds: number[];
  posSensorAdr: number[];
  velSensorAdr: number[];
  imuQuatAdr: number;
  imuGyroAdr: number;
  imuAccAdr: number;
  /** Per limb: sensordata address of its touch sensor, or -1. */
  touchAdr: number[];
  /** Per limb: tip site id, or -1. */
  tipSiteIds: number[];
  rangeAdr: number[];
  effort: number[];
}

/** Sensor readings available to controllers in sensor-only mode. */
export interface SensorReadings {
  /** Joint encoder positions (rad or m). */
  q: Float64Array;
  /** Joint encoder velocities. */
  qd: Float64Array;
  /** IMU orientation [w, x, y, z] in the world frame. */
  imuQuat: Quat;
  /** IMU angular velocity in the IMU frame, rad/s. */
  gyro: Vec3;
  /** IMU specific force in the IMU frame (includes gravity), m/s^2. */
  accel: Vec3;
  /** Per limb normal contact force at the tip (N), NaN where the limb has no touch sensor. */
  touch: number[];
  /** Range ray distances (m), -1 when nothing is hit. */
  ranges: number[];
}

function idOf(mj: MainModule, model: MjModel, type: number, name: string): number {
  const id = mj.mj_name2id(model, type, name);
  if (id < 0) throw new Error(`MuJoCo element "${name}" not found`);
  return id;
}

/**
 * Owns one MjModel/MjData pair built from a robot description. Every embind
 * object created here is deleted exactly once in dispose().
 */
export class MujocoRuntime {
  readonly layout: RuntimeLayout;
  readonly timestep: number;
  private disposedFlag = false;
  private initialQ: number[];
  private initialBase: Pose;

  private constructor(
    readonly mj: MainModule,
    readonly model: MjModel,
    readonly data: MjData,
    readonly description: RobotDescription,
    timestep: number,
    initialQ: number[],
  ) {
    this.timestep = timestep;
    this.layout = MujocoRuntime.resolveLayout(mj, model, description);
    this.initialQ = initialQ;
    this.initialBase = description.base.pose;
    this.reset();
  }

  static create(mj: MainModule, desc: RobotDescription, opts: MjcfOptions & { initialQ?: number[] } = {}): MujocoRuntime {
    const xml = generateMjcf(desc, opts);
    const model = mj.MjModel.from_xml_string(xml);
    let data: MjData | null = null;
    try {
      data = new mj.MjData(model);
      const dof = desc.limbs.reduce((s, l) => s + l.joints.length, 0);
      const q0 = opts.initialQ ?? new Array<number>(dof).fill(0);
      if (q0.length !== dof || !q0.every(Number.isFinite)) throw new RangeError(`Initial joint vector must have ${dof} finite values`);
      return new MujocoRuntime(mj, model, data, desc, opts.timestep ?? 0.002, q0);
    } catch (e) {
      data?.delete();
      model.delete();
      throw e;
    }
  }

  private static resolveLayout(mj: MainModule, model: MjModel, desc: RobotDescription): RuntimeLayout {
    const T = mj.mjtObj;
    const JOINT = T.mjOBJ_JOINT.value, BODY = T.mjOBJ_BODY.value, ACT = T.mjOBJ_ACTUATOR.value;
    const SENSOR = T.mjOBJ_SENSOR.value, SITE = T.mjOBJ_SITE.value;
    const sensorAdr = (name: string) => model.sensor_adr[idOf(mj, model, SENSOR, name)] as number;
    const floating = desc.base.kind === 'floating';
    const rootId = floating ? idOf(mj, model, JOINT, mjcfNames.root) : -1;
    const joints = desc.limbs.flatMap((l) => l.joints);
    const jointIdx = joints.map((j) => idOf(mj, model, JOINT, mjcfNames.joint(j.id)));
    return {
      nq: model.nq,
      nv: model.nv,
      nu: model.nu,
      floating,
      rootQposAdr: floating ? (model.jnt_qposadr[rootId] as number) : -1,
      rootDofAdr: floating ? (model.jnt_dofadr[rootId] as number) : -1,
      baseBodyId: idOf(mj, model, BODY, mjcfNames.base),
      jointIds: joints.map((j) => j.id),
      qposAdr: jointIdx.map((i) => model.jnt_qposadr[i] as number),
      dofAdr: jointIdx.map((i) => model.jnt_dofadr[i] as number),
      actuatorIndex: joints.map((j) => idOf(mj, model, ACT, mjcfNames.actuator(j.id))),
      linkBodyIds: joints.map((j) => idOf(mj, model, BODY, mjcfNames.linkBody(j.id))),
      posSensorAdr: joints.map((j) => sensorAdr(mjcfNames.posSensor(j.id))),
      velSensorAdr: joints.map((j) => sensorAdr(mjcfNames.velSensor(j.id))),
      imuQuatAdr: sensorAdr(mjcfNames.imuQuat),
      imuGyroAdr: sensorAdr(mjcfNames.imuGyro),
      imuAccAdr: sensorAdr(mjcfNames.imuAcc),
      touchAdr: desc.limbs.map((l) => (l.role === 'leg' && l.tip.radius > 0 ? sensorAdr(mjcfNames.touchSensor(l.id)) : -1)),
      tipSiteIds: desc.limbs.map((l) => (l.role === 'wheel' ? -1 : idOf(mj, model, SITE, mjcfNames.tipSite(l.id)))),
      rangeAdr: [],
      effort: joints.map((j) => j.limit.effort),
    };
  }

  /** Resolves range sensors by name (only present when generated with rangeRays). */
  attachRangeSensors(names: string[]): void {
    const SENSOR = this.mj.mjtObj.mjOBJ_SENSOR.value;
    this.layout.rangeAdr = names.map((n) => this.model.sensor_adr[idOf(this.mj, this.model, SENSOR, mjcfNames.rangeSensor(n))] as number);
  }

  get disposed(): boolean {
    return this.disposedFlag;
  }

  get time(): number {
    return this.data.time;
  }

  private assertLive(): void {
    if (this.disposedFlag) throw new Error('MujocoRuntime used after dispose()');
  }

  /** Sets joint positions (and the base pose for floating robots), zero velocity, then runs mj_forward. */
  setState(q: ArrayLike<number>, basePose: Pose = this.initialBase): void {
    this.assertLive();
    const L = this.layout;
    if (q.length !== L.jointIds.length) throw new RangeError('Joint vector size mismatch');
    const qpos = this.data.qpos as Float64Array;
    const qvel = this.data.qvel as Float64Array;
    qvel.fill(0);
    if (L.floating) {
      const quat = rpyToQuat(basePose.rpy);
      qpos.set([...basePose.xyz, ...quat], L.rootQposAdr);
    }
    for (let i = 0; i < L.qposAdr.length; i++) {
      const v = q[i]!;
      if (!Number.isFinite(v)) throw new RangeError(`Joint value ${i} is not finite`);
      qpos[L.qposAdr[i]!] = v;
    }
    (this.data.ctrl as Float64Array).fill(0);
    (this.data.xfrc_applied as Float64Array).fill(0);
    this.mj.mj_forward(this.model, this.data);
  }

  setInitialState(q: number[], basePose?: Pose): void {
    this.initialQ = q.slice();
    if (basePose) this.initialBase = basePose;
  }

  reset(): void {
    this.assertLive();
    this.mj.mj_resetData(this.model, this.data);
    this.setState(this.initialQ, this.initialBase);
  }

  step(): void {
    this.mj.mj_step(this.model, this.data);
  }

  /**
   * One closed-loop step using MuJoCo's split pipeline: mj_step1 computes the
   * current state and position/velocity sensors, `control` maps those readings
   * to torques, then mj_step2 applies them and integrates. Encoder and gyro
   * readings are therefore current; touch and accelerometer readings (which
   * need contact forces) come from the previous step.
   */
  controlStep(control: (s: SensorReadings) => ArrayLike<number> | null): SensorReadings {
    this.mj.mj_step1(this.model, this.data);
    const s = this.readSensors();
    const tau = control(s);
    if (tau) this.setTorques(tau);
    this.mj.mj_step2(this.model, this.data);
    return s;
  }

  /** Writes torques (N*m) to the motor actuators; MuJoCo clamps again to ctrlrange. */
  setTorques(tau: ArrayLike<number>): void {
    const ctrl = this.data.ctrl as Float64Array;
    const idx = this.layout.actuatorIndex;
    for (let i = 0; i < idx.length; i++) ctrl[idx[i]!] = Number.isFinite(tau[i]) ? tau[i]! : 0;
  }

  readSensors(): SensorReadings {
    const L = this.layout;
    const s = this.data.sensordata as Float64Array;
    const n = L.jointIds.length;
    const q = new Float64Array(n), qd = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      q[i] = s[L.posSensorAdr[i]!]!;
      qd[i] = s[L.velSensorAdr[i]!]!;
    }
    const a = L.imuQuatAdr, g = L.imuGyroAdr, c = L.imuAccAdr;
    return {
      q,
      qd,
      imuQuat: [s[a]!, s[a + 1]!, s[a + 2]!, s[a + 3]!],
      gyro: [s[g]!, s[g + 1]!, s[g + 2]!],
      accel: [s[c]!, s[c + 1]!, s[c + 2]!],
      touch: L.touchAdr.map((adr) => (adr >= 0 ? s[adr]! : Number.NaN)),
      ranges: L.rangeAdr.map((adr) => s[adr]!),
    };
  }

  /** Ground truth (simulator state, not a sensor): world transform of a body from xpos/xmat. */
  bodyTransform(bodyId: number): Mat4 {
    const xpos = this.data.xpos as Float64Array;
    const xmat = this.data.xmat as Float64Array;
    return fromRotationTranslation(xmat.subarray(9 * bodyId, 9 * bodyId + 9), [xpos[3 * bodyId]!, xpos[3 * bodyId + 1]!, xpos[3 * bodyId + 2]!]);
  }

  /** Ground truth: T_W_B of the base. */
  groundTruthBaseTransform(): Mat4 {
    return this.bodyTransform(this.layout.baseBodyId);
  }

  /** Ground truth: T_W_Li for every joint link, in global joint order. */
  groundTruthLinkTransforms(): Mat4[] {
    return this.layout.linkBodyIds.map((id) => this.bodyTransform(id));
  }

  /** Ground truth: world position of every limb tip site ([NaN x3] for wheels). */
  groundTruthTipPositions(): Vec3[] {
    const sx = this.data.site_xpos as Float64Array;
    return this.layout.tipSiteIds.map((id) => (id >= 0 ? [sx[3 * id]!, sx[3 * id + 1]!, sx[3 * id + 2]!] : [Number.NaN, Number.NaN, Number.NaN]));
  }

  /** Ground truth: world position of the centre of mass of the base and everything attached to it. */
  groundTruthCenterOfMass(): Vec3 {
    const c = this.data.subtree_com as Float64Array;
    const b = this.layout.baseBodyId;
    return [c[3 * b]!, c[3 * b + 1]!, c[3 * b + 2]!];
  }

  /** Ground truth: base position and tilt angle from vertical (rad). */
  groundTruthBaseState(): { pos: Vec3; tilt: number; linVel: Vec3 } {
    const T = this.groundTruthBaseTransform();
    const tilt = Math.acos(Math.max(-1, Math.min(1, T[10]!)));
    let linVel: Vec3 = [0, 0, 0];
    if (this.layout.floating) {
      const qvel = this.data.qvel as Float64Array;
      const a = this.layout.rootDofAdr;
      linVel = [qvel[a]!, qvel[a + 1]!, qvel[a + 2]!];
    }
    return { pos: [T[12]!, T[13]!, T[14]!], tilt, linVel };
  }

  /**
   * Ground truth: parts other than the feet within marginM of the floor
   * (z = 0): "body" when the lowest corner of the base box is, otherwise the
   * limb and joint names of a joint whose link capsules reach that low.
   */
  groundTruthTouchingParts(marginM = 0.005): string[] {
    const desc = this.description;
    const out: string[] = [];
    const T = this.groundTruthBaseTransform();
    const [sx, sy, sz] = desc.base.size;
    if (T[14]! - (Math.abs(T[2]!) * sx + Math.abs(T[6]!) * sy + Math.abs(T[10]!) * sz) / 2 < marginM) out.push('body');
    const links = this.groundTruthLinkTransforms();
    let j = 0;
    for (const limb of desc.limbs) {
      limb.joints.forEach((joint, k) => {
        const z = links[j++]![14]!;
        if (limb.role === 'wheel') return;
        // A joint is where its parent link capsule ends and its own begins.
        const r = Math.max(joint.link.radius, k > 0 ? limb.joints[k - 1]!.link.radius : 0);
        if (z - r < marginM) out.push(`${limb.name.toLowerCase()} ${joint.name.toLowerCase()}`);
      });
    }
    return out;
  }

  /** Applies a world-frame force (N) and torque (N*m) at the centre of mass of a body until cleared. */
  setBodyWrench(bodyId: number, force: Vec3, torque: Vec3 = [0, 0, 0]): void {
    const x = this.data.xfrc_applied as Float64Array;
    x.set([...force, ...torque], 6 * bodyId);
  }

  clearWrenches(): void {
    (this.data.xfrc_applied as Float64Array).fill(0);
  }

  /** Bias forces (gravity plus Coriolis and centrifugal) at the actuated joints from the last forward pass. */
  biasForces(): number[] {
    this.assertLive();
    const bias = this.data.qfrc_bias as Float64Array;
    return this.layout.dofAdr.map((adr) => bias[adr]!);
  }

  hasNonFiniteState(): boolean {
    const qpos = this.data.qpos as Float64Array;
    const qvel = this.data.qvel as Float64Array;
    for (let i = 0; i < qpos.length; i++) if (!Number.isFinite(qpos[i])) return true;
    for (let i = 0; i < qvel.length; i++) if (!Number.isFinite(qvel[i])) return true;
    return false;
  }

  /** Counts of MuJoCo warnings (bad qacc, contact overflow, ...) since the last reset. */
  warningCounts(): Record<string, number> {
    // The vector is owned by MjData (deleting it aborts the module); elements returned by get() are copies.
    const vec = this.data.warning;
    const out: Record<string, number> = {};
    const W = this.mj.mjtWarning as unknown as Record<string, { value: number } | undefined>;
    for (const label of ['inertia', 'contactfull', 'cnstrfull', 'badqpos', 'badqvel', 'badqacc', 'badctrl']) {
      const entry = W[`mjWARN_${label.toUpperCase()}`];
      if (!entry || typeof entry.value !== 'number') continue;
      const stat = vec.get(entry.value);
      if (!stat) continue;
      out[label] = stat.number;
      stat.delete();
    }
    return out;
  }

  /** WASM linear memory size in bytes (grows, never shrinks). */
  wasmMemoryBytes(): number {
    return (this.data.qpos as Float64Array).buffer.byteLength;
  }

  /**
   * Diagonal of the joint-space inertia matrix (kg*m^2, including armature) for
   * each actuated joint at the current configuration, via M * e_i (mj_mulM).
   */
  jointInertias(): number[] {
    const nv = this.model.nv;
    const res = new this.mj.DoubleBuffer(nv);
    try {
      return this.layout.dofAdr.map((adr) => {
        const e = new Array<number>(nv).fill(0);
        e[adr] = 1;
        this.mj.mj_mulM(this.model, this.data, res, e);
        return (res.GetView() as Float64Array)[adr]!;
      });
    } finally {
      res.delete();
    }
  }

  dispose(): void {
    if (this.disposedFlag) return;
    this.disposedFlag = true;
    this.data.delete();
    this.model.delete();
  }
}
