import { pdControl } from '../../src/core/control/pd';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import { loadMujocoBrowser } from '../../src/physics/loader';
import { MujocoRuntime } from '../../src/physics/runtime';
import { RobotView } from '../../src/render/robotView';
import { Viewport } from '../../src/render/viewport';

const statsEl = document.getElementById('stats')!;
const mj = await loadMujocoBrowser();
const desc = quadruped();
const qCmd = quadrupedStance();
const rt = MujocoRuntime.create(mj, desc, { initialQ: qCmd });
const gains = qCmd.map(() => ({ kp: 40, kd: 1.0 }));
const viewport = new Viewport(document.getElementById('view')!);
const view = new RobotView(desc);
viewport.scene.add(view.root);

let pushRemaining = 0;
let acc = 0;
const stepTimes: number[] = [];
const frameTimes: number[] = [];
let minHeightAfterPush = Infinity;
let maxTiltAfterPush = 0;

function tick(): void {
  if (pushRemaining > 0) {
    rt.setBodyWrench(rt.layout.baseBodyId, [0, 40, 0]);
    pushRemaining -= rt.timestep;
  } else {
    rt.clearWrenches();
  }
  rt.controlStep((s) => pdControl(gains, rt.layout.effort, { qCmd, qMeas: s.q, qdMeas: s.qd }).tauApplied);
}

document.getElementById('push')!.addEventListener('click', () => {
  pushRemaining = 0.05;
  minHeightAfterPush = Infinity;
  maxTiltAfterPush = 0;
});
document.getElementById('reset')!.addEventListener('click', () => rt.reset());

viewport.start((dt) => {
  const f0 = performance.now();
  acc += dt;
  let n = 0;
  const t0 = performance.now();
  while (acc >= rt.timestep && n < 25) {
    tick();
    acc -= rt.timestep;
    n++;
  }
  if (n === 25) acc = 0; // Drop excess time instead of spiralling when frames are slow.
  if (n > 0) stepTimes.push((performance.now() - t0) / n);
  if (stepTimes.length > 600) stepTimes.shift();
  view.applyTransforms(rt.groundTruthBaseTransform(), rt.groundTruthLinkTransforms());
  const b = rt.groundTruthBaseState();
  minHeightAfterPush = Math.min(minHeightAfterPush, b.pos[2]);
  maxTiltAfterPush = Math.max(maxTiltAfterPush, b.tilt);
  frameTimes.push(performance.now() - f0);
  if (frameTimes.length > 600) frameTimes.shift();
  const sorted = stepTimes.slice().sort((a, c) => a - c);
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  const result = {
    simTimeS: +rt.time.toFixed(2),
    baseHeightM: +b.pos[2].toFixed(4),
    tiltRad: +b.tilt.toFixed(4),
    stepMsMedian: +median.toFixed(4),
    stepsPerFrame: n,
    minHeightSincePushM: +minHeightAfterPush.toFixed(4),
    maxTiltSincePushRad: +maxTiltAfterPush.toFixed(4),
    nonFinite: rt.hasNonFiniteState(),
  };
  (window as unknown as { __spike: unknown }).__spike = result;
  statsEl.textContent = Object.entries(result).map(([k, v]) => `${k.padEnd(22)} ${v}`).join('\n');
});
