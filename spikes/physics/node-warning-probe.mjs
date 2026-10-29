// Diagnoses which embind accessors return copies that must be deleted.
import loadMujoco from '@mujoco/mujoco';

const mj = await loadMujoco();
const xml = `<mujoco><worldbody><geom type="plane" size="1 1 0.1"/>
<body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>`;

function trial(label, fn) {
  const m = mj.MjModel.from_xml_string(xml);
  const d = new mj.MjData(m);
  try {
    fn(m, d);
    d.delete();
    m.delete();
    console.log(label, 'ok');
  } catch (e) {
    console.log(label, 'FAILED', e.message);
  }
}

trial('no warning access', () => {});
trial('warning access without delete', (m, d) => { const w = d.warning; console.log(' size', w.size(), 'isAliasOf?', typeof w.isAliasOf); });
trial('warning get without delete', (m, d) => { const w = d.warning; const s = w.get(0); console.log(' number', s.number); });
trial('warning vec delete only', (m, d) => { const w = d.warning; w.delete(); });
trial('warning get + stat delete + vec delete', (m, d) => { const w = d.warning; const s = w.get(0); s.delete(); w.delete(); });
