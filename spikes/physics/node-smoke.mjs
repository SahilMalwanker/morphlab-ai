// Headless smoke test: load the official MuJoCo WASM build in Node and step a falling sphere.
import loadMujoco from '@mujoco/mujoco';

const t0 = performance.now();
const mj = await loadMujoco();
const t1 = performance.now();
const xml = `<mujoco><compiler angle="radian"/><option timestep="0.002"/>
<worldbody><geom type="plane" size="1 1 0.1"/>
<body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>`;
const m = mj.MjModel.from_xml_string(xml);
const d = new mj.MjData(m);
try {
  for (let i = 0; i < 500; i++) mj.mj_step(m, d);
  console.log('load ms', (t1 - t0).toFixed(0), 'nq', m.nq, 'nv', m.nv, 'time', d.time.toFixed(3),
    'z', d.qpos[2].toFixed(4), 'qpos type', d.qpos.constructor.name, 'ncon', d.ncon);
  console.log('warning field type', typeof d.warning, d.warning?.constructor?.name);
  console.log('HEAP keys', Object.keys(mj).filter((k) => k.startsWith('HEAP')).join(','));
} finally {
  d.delete();
  m.delete();
}
