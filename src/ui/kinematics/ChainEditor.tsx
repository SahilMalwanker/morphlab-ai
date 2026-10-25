import { useEffect, useState } from 'react';
import { joint } from '../../core/model/presets';
import type { RobotDescription } from '../../core/model/types';
import { validateRobot } from '../../core/model/validate';
import { DEG2RAD, RAD2DEG } from '../components';

type Axis = 'x' | 'y' | 'z';
export interface Row {
  axis: Axis;
  length: string;
  lower: string;
  upper: string;
}

const AXES: Record<Axis, [number, number, number]> = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };

/** True for fixed-base single-arm chains whose links extend along +Z at q = 0. */
export function isEditableChain(d: RobotDescription): boolean {
  if (d.base.kind !== 'fixed' || d.limbs.length !== 1) return false;
  const l = d.limbs[0]!;
  if (l.role !== 'arm') return false;
  const alongZ = (v: number[]) => Math.abs(v[0]!) < 1e-12 && Math.abs(v[1]!) < 1e-12 && v[2]! > 0;
  return (
    l.joints.every((j, i) => j.type === 'revolute' && j.origin.rpy.every((r) => r === 0) && (i === 0 || alongZ(j.origin.xyz))) &&
    l.joints.every((j) => j.axis.filter((c) => c !== 0).length === 1) &&
    alongZ(l.tip.xyz)
  );
}

function rowsFrom(d: RobotDescription): Row[] {
  const l = d.limbs[0]!;
  return l.joints.map((j, i) => {
    const next = i + 1 < l.joints.length ? l.joints[i + 1]!.origin.xyz[2] : l.tip.xyz[2];
    const axis: Axis = j.axis[0] !== 0 ? 'x' : j.axis[1] !== 0 ? 'y' : 'z';
    return { axis, length: next.toFixed(3), lower: (j.limit.lower * RAD2DEG).toFixed(0), upper: (j.limit.upper * RAD2DEG).toFixed(0) };
  });
}

export function buildChain(base: RobotDescription, rows: Row[]): { desc?: RobotDescription; errors: string[] } {
  const errors: string[] = [];
  const nums = rows.map((r, i) => {
    const length = Number(r.length), lower = Number(r.lower), upper = Number(r.upper);
    if (!(Number.isFinite(length) && length >= 0.02 && length <= 1)) errors.push(`Joint ${i + 1}: link length must be between 0.02 and 1 m.`);
    if (!(Number.isFinite(lower) && Number.isFinite(upper) && lower >= -180 && upper <= 180)) errors.push(`Joint ${i + 1}: limits must be between -180 and 180 degrees.`);
    else if (!(lower < upper)) errors.push(`Joint ${i + 1}: lower limit must be below upper limit.`);
    return { length, lower: lower * DEG2RAD, upper: upper * DEG2RAD };
  });
  if (errors.length > 0) return { errors };
  const first = base.limbs[0]!.joints[0]!;
  const joints = rows.map((r, i) =>
    joint({
      id: `j${i + 1}`,
      name: `Joint ${i + 1} (${r.axis.toUpperCase()})`,
      xyz: i === 0 ? first.origin.xyz : [0, 0, nums[i - 1]!.length],
      axis: AXES[r.axis],
      lower: nums[i]!.lower,
      upper: nums[i]!.upper,
      effort: 20,
      radius: Math.max(0.015, 0.035 - 0.004 * i),
      mass: Math.max(0.15, 0.7 - 0.1 * i),
    }),
  );
  const desc: RobotDescription = {
    ...base,
    id: 'custom_chain',
    name: `Custom ${rows.length}-joint chain`,
    limbs: [{ ...base.limbs[0]!, joints, tip: { xyz: [0, 0, nums[nums.length - 1]!.length], radius: 0.02 } }],
  };
  const issues = validateRobot(desc);
  return issues.length ? { errors: issues.map((x) => x.message) } : { desc, errors: [] };
}

export function ChainEditor({ description, onApply }: { description: RobotDescription; onApply: (d: RobotDescription) => void }) {
  const [rows, setRows] = useState<Row[]>(() => rowsFrom(description));
  const [errors, setErrors] = useState<string[]>([]);
  useEffect(() => {
    setRows(rowsFrom(description));
    setErrors([]);
  }, [description]);
  const set = (i: number, patch: Partial<Row>) => setRows((rs) => rs.map((r, k) => (k === i ? { ...r, ...patch } : r)));
  return (
    <div>
      <table className="chain">
        <thead>
          <tr><th>#</th><th>Axis</th><th>Link (m)</th><th>Min (deg)</th><th>Max (deg)</th></tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="muted">{i + 1}</td>
              <td>
                <select aria-label={`Joint ${i + 1} axis`} value={r.axis} onChange={(e) => set(i, { axis: e.target.value as Axis })}>
                  <option value="x">X</option><option value="y">Y</option><option value="z">Z</option>
                </select>
              </td>
              <td><input aria-label={`Joint ${i + 1} link length`} type="number" step={0.01} value={r.length} onChange={(e) => set(i, { length: e.target.value })} /></td>
              <td><input aria-label={`Joint ${i + 1} lower limit`} type="number" step={5} value={r.lower} onChange={(e) => set(i, { lower: e.target.value })} /></td>
              <td><input aria-label={`Joint ${i + 1} upper limit`} type="number" step={5} value={r.upper} onChange={(e) => set(i, { upper: e.target.value })} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: 8 }}>
        <button className="btn" disabled={rows.length >= 7} onClick={() => setRows((rs) => [...rs, { axis: 'y', length: '0.150', lower: '-150', upper: '150' }])}>Add joint</button>
        <button className="btn" disabled={rows.length <= 1} onClick={() => setRows((rs) => rs.slice(0, -1))}>Remove last</button>
        <button
          className="btn primary"
          onClick={() => {
            const r = buildChain(description, rows);
            setErrors(r.errors);
            if (r.desc) onApply(r.desc);
          }}
        >
          Apply
        </button>
      </div>
      {errors.map((e) => (
        <p key={e} className="note error" role="alert">{e}</p>
      ))}
      <p className="note muted">Links extend along each joint frame's Z axis at zero angle. Up to 7 joints.</p>
    </div>
  );
}
