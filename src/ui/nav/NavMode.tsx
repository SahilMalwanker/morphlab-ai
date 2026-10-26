import { useEffect, useId, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type { LongTermMemory } from '../../ai/memory';
import { PRESETS, type PresetId } from '../../core/model/catalog';
import type { RobotDescription } from '../../core/model/types';
import { driveGeometry, LEG_RAY_HEIGHT_M } from '../../nav/drive';
import { RAY_ANGLES, RAY_MAX_RANGE } from '../../nav/stack';
import { ARENA, OBSTACLE_HEIGHT, TEST_MAPS } from '../../nav/maps';
import { RobotView } from '../../render/robotView';
import { disposeObject, Viewport } from '../../render/viewport';
import { NumberField, Readout, Section, StatusBadge } from '../components';
import { Explain } from '../learn';
import { useStore, useVersion } from '../store';
import { DRIVE_KEYS, isDrivingKey } from './keys';
import { rememberPlace } from './places';
import type { NavSession } from './session';

const INSET = { w: 300, h: 180 };
const OVERLAY_LAYER = 1;

function NavViewport({ session, memory }: { session: NavSession; memory: LongTermMemory }) {
  const host = useRef<HTMLDivElement>(null);
  const mapId = useStore(session.store).mapId;
  // setRobot replaces the description object, so its identity marks a robot change.
  const robot = session.robot;
  useEffect(() => {
    const vp = new Viewport(host.current!, { cameraPosition: [1.5, -3.2, 3.0], target: [1.5, 0, 0] });
    const map = TEST_MAPS.find((m) => m.id === mapId)!;
    const world = new THREE.Group();
    vp.scene.add(world);
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x6e7681, roughness: 0.8 });
    for (const b of map.boxes) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(2 * b.hx, 2 * b.hy, OBSTACLE_HEIGHT), boxMat);
      m.position.set(b.cx, b.cy, OBSTACLE_HEIGHT / 2);
      m.castShadow = true;
      world.add(m);
    }
    const goal = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.2, 32), new THREE.MeshBasicMaterial({ color: 0x3fb950, side: THREE.DoubleSide }));
    goal.position.z = 0.004;
    world.add(goal);
    const path = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x58a6ff }));
    path.position.z = 0.01;
    world.add(path);
    const rays = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xe3b341, transparent: true, opacity: 0.8 }));
    const rayPos = new THREE.BufferAttribute(new Float32Array(RAY_ANGLES.length * 6), 3);
    rays.geometry.setAttribute('position', rayPos);
    rays.geometry.setDrawRange(0, 0);
    world.add(rays);
    // Both geometries are replaced while driving; a cached bounding sphere would cull them.
    path.frustumCulled = false;
    rays.frustumCulled = false;
    // Occupancy grid as a texture over the arena floor.
    const spec = { w: 84, h: 84 };
    const pixels = new Uint8Array(spec.w * spec.h * 4);
    const tex = new THREE.DataTexture(pixels, spec.w, spec.h, THREE.RGBAFormat);
    tex.magFilter = THREE.NearestFilter;
    const gridPlane = new THREE.Mesh(new THREE.PlaneGeometry(ARENA.maxX - ARENA.minX, ARENA.maxY - ARENA.minY), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    gridPlane.position.set((ARENA.minX + ARENA.maxX) / 2, (ARENA.minY + ARENA.maxY) / 2, 0.003);
    world.add(gridPlane);
    const view = new RobotView(robot);
    vp.scene.add(view.root);
    // Remembered places (world memory) as small markers; redrawn when memory changes.
    const placeGeom = new THREE.CylinderGeometry(0.06, 0.06, 0.012, 24).rotateX(Math.PI / 2);
    const placeMat = new THREE.MeshBasicMaterial({ color: 0xd2a8ff });
    const places = new THREE.Group();
    places.position.z = 0.008;
    world.add(places);
    const drawPlaces = () => {
      places.clear();
      for (const p of memory.places(map.id)) {
        const marker = new THREE.Mesh(placeGeom, placeMat);
        marker.position.set(p.x, p.y, 0);
        marker.layers.set(OVERLAY_LAYER);
        places.add(marker);
      }
    };
    drawPlaces();
    const unsubscribePlaces = memory.subscribe(drawPlaces);
    // Overlays live on layer 1: the main camera shows them, the virtual camera does not.
    for (const o of [goal, path, rays, gridPlane, places]) o.layers.set(OVERLAY_LAYER);
    vp.camera.layers.enable(OVERLAY_LAYER);
    // Virtual camera just ahead of the robot's front, drawn as an inset.
    const cam = new THREE.PerspectiveCamera(70, INSET.w / INSET.h, 0.02, 20);
    cam.up.set(0, 0, 1);
    const clear = new THREE.Color();
    const ahead = session.drive.rayOrigin[0] + 0.02;
    const aimCamera = (T: ArrayLike<number>) => {
      const yaw = Math.atan2(T[1]!, T[0]!);
      cam.position.set(T[12]! + ahead * Math.cos(yaw), T[13]! + ahead * Math.sin(yaw), T[14]! + 0.08);
      cam.lookAt(cam.position.x + Math.cos(yaw), cam.position.y + Math.sin(yaw), cam.position.z - 0.15);
    };
    vp.afterRender = (renderer, w, h) => {
      const rt = session.runtime;
      // The inset takes at most 40 % of the viewport width so it never hides the main view.
      const iw = Math.round(Math.min(INSET.w, w * 0.4)), ih = Math.round((iw * INSET.h) / INSET.w);
      if (!rt || iw < 120) return;
      aimCamera(rt.groundTruthBaseTransform());
      const x = w - iw - 12, y = 12;
      renderer.getClearColor(clear);
      const alpha = renderer.getClearAlpha();
      renderer.setScissorTest(true);
      renderer.setScissor(x - 1, y - 1, iw + 2, ih + 2);
      renderer.setClearColor(0x6e7681, 1);
      renderer.clear();
      renderer.setClearColor(clear, alpha);
      renderer.setScissor(x, y, iw, ih);
      renderer.setViewport(x, y, iw, ih);
      renderer.render(vp.scene, cam);
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, w, h);
    };
    let lastGrid = 0;
    let lastPath: unknown = null;
    vp.start((dt) => {
      session.advance(dt);
      const rt = session.runtime;
      if (!rt) return;
      view.applyTransforms(rt.groundTruthBaseTransform(), rt.groundTruthLinkTransforms());
      const g = session.state.goal;
      goal.position.x = g[0];
      goal.position.y = g[1];
      const segs = session.raySegments();
      segs.forEach(([a, b, c, d], i) => (rayPos.array as Float32Array).set([a, b, 0.08, c, d, 0.08], i * 6));
      rayPos.needsUpdate = true;
      rays.geometry.setDrawRange(0, segs.length * 2);
      const stack = session.navStack;
      if (stack && stack.path !== lastPath) {
        lastPath = stack.path;
        path.geometry.dispose();
        path.geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(stack.path.flatMap(([x, y]) => [x, y, 0]), 3));
      }
      const now = performance.now();
      if (stack && now - lastGrid > 200) {
        lastGrid = now;
        const lo = stack.grid.logOdds;
        for (let i = 0; i < lo.length; i++) {
          const v = lo[i]!;
          pixels[4 * i] = v > 0 ? 240 : 40;
          pixels[4 * i + 1] = v > 0 ? 97 : 160;
          pixels[4 * i + 2] = v > 0 ? 109 : 100;
          pixels[4 * i + 3] = v > 0.85 ? 200 : v < -0.4 ? 50 : 0;
        }
        tex.needsUpdate = true;
      }
    });
    // Double-click on the floor sets the goal.
    const ray = new THREE.Raycaster();
    const floor = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const onDbl = (e: MouseEvent) => {
      const r = vp.renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), vp.camera);
      const p = new THREE.Vector3();
      if (ray.ray.intersectPlane(floor, p)) session.setGoal(p.x, p.y);
    };
    vp.renderer.domElement.addEventListener('dblclick', onDbl);
    return () => {
      unsubscribePlaces();
      vp.renderer.domElement.removeEventListener('dblclick', onDbl);
      view.dispose();
      tex.dispose();
      placeGeom.dispose();
      placeMat.dispose();
      disposeObject(world);
      vp.dispose();
    };
  }, [session, mapId, memory, robot]);
  return (
    <div className="viewport" ref={host} data-testid="viewport">
      <div className="viewport-label">Navigate: MuJoCo simulation. Rays, map and path come from the controller; the robot is drawn from simulator state; purple markers are remembered places. Inset: virtual camera (display only).</div>
    </div>
  );
}

/**
 * Keyboard driving while the Navigate tab is open: a drive key takes over
 * (also from Drive to goal) and releasing every key stops the rover. Which
 * key events drive is decided by `isDrivingKey` (never during text entry or
 * with modifiers; on tabs and sliders only W A S D). Driving stops when the
 * window loses focus or the page is hidden, so a held key can never keep
 * the rover moving unattended.
 */
function useKeyboardDriving(session: NavSession) {
  useEffect(() => {
    const down = new Set<string>();
    const apply = () => {
      const has = (k: string[]) => k.some((c) => down.has(c));
      const { remoteSpeed, remoteTurn } = session.limits;
      const v = remoteSpeed * ((has(DRIVE_KEYS.forward) ? 1 : 0) - (has(DRIVE_KEYS.back) ? 1 : 0));
      const w = remoteTurn * ((has(DRIVE_KEYS.left) ? 1 : 0) - (has(DRIVE_KEYS.right) ? 1 : 0));
      session.setRemote(v, w);
    };
    const halt = (why: string) => {
      down.clear();
      if (session.state.driving === 'remote') session.stop(`Remote control stopped: ${why}.`);
    };
    const onDown = (e: KeyboardEvent) => {
      if (session.state.status !== 'ready' || !isDrivingKey(e)) return;
      e.preventDefault();
      if (down.has(e.code)) return;
      down.add(e.code);
      apply();
    };
    const onUp = (e: KeyboardEvent) => {
      if (!down.delete(e.code)) return;
      // After Stop or End remote control, releasing a key must not start driving again.
      if (session.state.driving === 'remote') apply();
    };
    const onBlur = () => halt('the window lost focus');
    const onVisibility = () => {
      if (document.visibilityState === 'hidden') halt('the page was hidden');
    };
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    window.addEventListener('blur', onBlur);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup', onUp);
      window.removeEventListener('blur', onBlur);
      document.removeEventListener('visibilitychange', onVisibility);
      if (session.state.driving !== 'idle') session.stop();
    };
  }, [session]);
}

function SelectField({ label, value, options, onChange, testId }: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void; testId?: string }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)} data-testid={testId}>
        {options.map(([v, l]) => (
          <option key={v} value={v}>{l}</option>
        ))}
      </select>
    </div>
  );
}

interface RobotChoice {
  value: PresetId | 'custom' | 'loaded';
  label: string;
  robot: RobotDescription;
  /** Why the navigation stack cannot drive this body, or null. */
  problem: string | null;
}

/**
 * Every preset and the Build robot. Bodies Navigate cannot move stay listed
 * but disabled, and the reasons are shown in text.
 * A Build robot loaded earlier stays listed after Build changes it; the
 * current Build robot is then a separate choice.
 */
function RobotPicker({ session, buildRobot }: { session: NavSession; buildRobot: RobotDescription | null }) {
  const s = useStore(session.store);
  const id = useId();
  const reasonsId = useId();
  const loaded = s.robotSource === 'custom' ? session.robot : null;
  const stale = loaded !== null && loaded !== buildRobot;
  const choices = useMemo<RobotChoice[]>(() => {
    const out: RobotChoice[] = [];
    if (stale && loaded) out.push({ value: 'loaded', label: `Loaded from Build: ${loaded.name}`, robot: loaded, problem: null });
    if (buildRobot) {
      const r = driveGeometry(buildRobot);
      out.push({ value: 'custom', label: `Build robot: ${buildRobot.name}`, robot: buildRobot, problem: r.ok ? null : r.reason });
    }
    for (const p of PRESETS) {
      const robot = p.make();
      const r = driveGeometry(robot);
      out.push({ value: p.id, label: p.label, robot, problem: r.ok ? null : r.reason });
    }
    return out;
  }, [loaded, stale, buildRobot]);
  const blocked = choices.filter((c) => c.problem);
  const byReason = new Map<string, string[]>();
  for (const c of blocked) byReason.set(c.problem!, [...(byReason.get(c.problem!) ?? []), c.value === 'custom' ? `the Build robot (${c.robot.name})` : c.label]);
  const choose = (value: string) => {
    const c = choices.find((x) => x.value === value);
    if (!c || c.problem || c.value === 'loaded') return;
    if (c.value === 'custom') session.setRobot(c.robot, 'custom');
    else session.setPreset(c.value);
  };
  const value = s.robotSource === 'custom' ? (stale ? 'loaded' : 'custom') : s.robotSource;
  return (
    <>
      <div className="field">
        <label htmlFor={id}>Robot</label>
        <select id={id} value={value} onChange={(e) => choose(e.target.value)} aria-describedby={blocked.length ? reasonsId : undefined} data-testid="nav-robot">
          {choices.map((c) => (
            <option key={c.value} value={c.value} disabled={c.problem !== null}>
              {c.problem ? `${c.label} (cannot drive)` : c.label}
            </option>
          ))}
        </select>
      </div>
      {blocked.length > 0 && (
        <div className="note muted" id={reasonsId} data-testid="nav-robot-reasons">
          Navigate drives wheeled bodies by skid steering and walks bodies with four or more legs. Not drivable here:
          <ul style={{ margin: '4px 0 0', paddingLeft: 16 }}>
            {[...byReason].map(([reason, names]) => (
              <li key={reason}>{names.join(', ')}: {reason}.</li>
            ))}
          </ul>
        </div>
      )}
      {!buildRobot && <p className="note muted">Open Build to design a robot; a wheeled or legged one can then be driven here.</p>}
    </>
  );
}

const SOURCE_SHORT = { 'sensor-estimate': 'from the estimate', 'ground-truth': 'from ground truth', given: 'given' } as const;

/** World memory for the current map: named positions kept in this browser; the AI can use them by name. */
function Places({ session, memory }: { session: NavSession; memory: LongTermMemory }) {
  const s = useStore(session.store);
  useVersion(memory);
  const [name, setName] = useState('');
  const [note, setNote] = useState<{ error: boolean; text: string } | null>(null);
  useEffect(() => {
    setNote(null);
  }, [s.mapId]);
  const places = memory.places(s.mapId);
  const legged = session.drive.kind === 'legged';
  const remember = () => {
    const r = rememberPlace(session, memory, name);
    setNote(r.ok ? { error: false, text: `${r.replaced ? 'Updated' : 'Remembered'} "${r.place.name}" at (${r.place.x.toFixed(2)}, ${r.place.y.toFixed(2)}) m, ${SOURCE_SHORT[r.place.source]}.` } : { error: true, text: r.reason });
    if (r.ok) setName('');
  };
  return (
    <Section title={`Remembered places (${places.length})`}>
      <p className="note muted">Named positions on this map, kept in this browser. Remember stores where the controller believes the robot is ({s.mode === 'sensor-only' ? 'its sensor-only estimate' : 'ground truth in this mode'}). AI commands can use them by name.</p>
      <div className="row">
        <input type="text" value={name} maxLength={40} placeholder="dock" autoComplete="off" aria-label="Place name" onChange={(e) => setName(e.target.value)} style={{ flex: 1, minWidth: 0 }} data-testid="place-name" />
        <button className="btn" type="button" onClick={remember} disabled={s.status !== 'ready' || !name.trim()} data-testid="place-remember">Remember position</button>
      </div>
      {note && <p className={`note ${note.error ? 'error' : ''}`} role="status" data-testid="place-note">{note.text}</p>}
      {memory.error && <p className="note error" role="alert">{memory.error}</p>}
      {places.length > 0 && (
        <ul style={{ margin: '6px 0 0', paddingLeft: 16, fontSize: 12 }} data-testid="places">
          {places.map((p) => (
            <li key={p.id} style={{ marginBottom: 4 }}>
              <span className="mono">{p.name}</span>: ({p.x.toFixed(2)}, {p.y.toFixed(2)}) m, {SOURCE_SHORT[p.source]}{' '}
              <button className="btn" type="button" onClick={() => session.setGoal(p.x, p.y) && session.startAuto()} disabled={!session.drivable}>{legged ? 'Walk here' : 'Drive here'}</button>{' '}
              <button
                className="btn"
                type="button"
                onClick={() => {
                  memory.forgetPlace(p.mapId, p.name);
                  setNote(null);
                }}
                data-testid="place-forget"
              >
                Forget
              </button>
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}

export function NavMode({ session, memory, buildRobot }: { session: NavSession; memory: LongTermMemory; buildRobot: RobotDescription | null }) {
  const s = useStore(session.store);
  const [goalAttempt, setGoalAttempt] = useState(0);
  useEffect(() => {
    void session.init();
    // A walking controller trained in the AI Playground since the last visit walks here too.
    void session.refreshWalking();
  }, [session]);
  const remoteOn = s.driving === 'remote';
  useKeyboardDriving(session);
  const f = (v: number | undefined, d = 2) => (v === undefined || !Number.isFinite(v) ? 'n/a' : v.toFixed(d));
  const ev = s.evaluation;
  const reachedCount = ev.results.filter((r) => r.reached).length;
  const drive = session.drive;
  const legged = drive.kind === 'legged';
  const [lo, hi] = session.limits.speedLimitRange;
  return (
    <>
      <aside className="panel left" aria-label="Navigation setup">
        <Section title="Robot and map">
          <RobotPicker session={session} buildRobot={buildRobot} />
          <p className="note muted" data-testid="nav-robot-geometry">
            {drive.kind === 'wheeled'
              ? `${drive.wheels.length} wheels, radius ${f(drive.wheelRadius * 100, 1)} cm, track ${f(drive.trackWidth)} m, footprint ${f(2 * drive.halfExtents[0])} x ${f(2 * drive.halfExtents[1])} m (from its description).`
              : `${drive.legs.length} legs, trot gait${drive.legs.length > 4 ? ' (alternating tripods on six legs)' : ''} at ${drive.gait.cycleS} s cycles, up to ${f(hi)} m/s and ${f(session.limits.maxTurnRate, 1)} rad/s commanded; footprint ${f(2 * drive.halfExtents[0])} x ${f(2 * drive.halfExtents[1])} m with room for the steps; range rays ${f(LEG_RAY_HEIGHT_M)} m above the floor, ahead of the front feet (from its description).`}
          </p>
          {s.walker && (
            <>
              <p className="note" data-testid="nav-walker">
                Walks with {s.walker.text}
                {s.walkerFault ? ` The learned policy stopped (${s.walkerFault}); the calibrated generator walks on.` : ''}
              </p>
              {s.walkingJob && (
                <p className="note" role="status" data-testid="nav-train-status">
                  Training a walking controller: {Math.round(s.walkingJob.fraction * 100)} %. {s.walkingJob.detail}
                </p>
              )}
              <div className="row">
                {s.walkingJob ? (
                  <button className="btn" type="button" onClick={() => session.cancelWalkingTraining()} data-testid="nav-train-cancel">Cancel</button>
                ) : (
                  <button className="btn" type="button" onClick={() => void session.trainWalking()} disabled={s.status !== 'ready'} data-testid="nav-train-walking">
                    {s.walker.learned ? 'Train the walking controller again' : 'Train a walking controller'}
                  </button>
                )}
              </div>
              <p className="note muted">Training takes four to eight minutes in background workers: the gait generator is calibrated on measured walks and a feedback policy is learned from simulated runs, then both are evaluated. The AI trains it when you ask the robot to learn to walk.</p>
            </>
          )}
          <SelectField label="Test map" value={s.mapId} options={TEST_MAPS.map((m) => [m.id, m.name])} onChange={(v) => session.setMap(v)} testId="nav-map" />
          <SelectField label="Controller inputs" value={s.mode} options={[['sensor-only', `Sensor-only (${legged ? 'leg odometry' : 'encoders'}, IMU, range rays)`], ['ground-truth', 'Ground truth (simulator pose and known map)']]} onChange={(v) => session.setMode(v as 'sensor-only' | 'ground-truth')} testId="nav-mode" />
          <p className="note muted">{s.mode === 'sensor-only' ? `The controller estimates its pose from ${legged ? 'leg odometry (joint encoders and foot touch sensors)' : 'wheel encoders'} and the IMU and builds the map from nine range rays. It never receives simulator state.` : 'Ground-truth mode: the controller is given the simulator pose and the true obstacle map. Use it as a reference, not as perception.'}</p>
          <Explain>
            {legged ? (
              <p>Leg odometry: a foot on the ground does not move, so the body moves opposite to how the feet move under it, computed from the joint encoders; a foot counts as on the ground while its touch sensor reads more than 1 N. The IMU gives the heading. Slipping feet make the estimate drift, and nothing corrects it; compare the two position readouts. The rays bob with the walking body, so a hit the IMU attitude puts on the floor is not mapped as an obstacle.</p>
            ) : (
              <p>Odometry integrates the distance from the wheel encoders along the heading from the IMU. Nothing corrects it, so its error only grows; compare the two position readouts.</p>
            )}
            <p>Each range ray lowers the occupied probability (log-odds) of the cells it passes and raises it for the cell where it hits something. Green cells are probably free, red cells probably occupied, the rest unknown.</p>
            <p>A* searches the grid for the cheapest path. Cells within {drive.robotRadiusM.toFixed(2)} m of an occupied cell are blocked (half the robot's footprint diagonal plus a margin) and cells within {Math.max(0.75, drive.robotRadiusM + 0.4).toFixed(2)} m cost extra, so paths keep clear of walls. The plan is redone every 0.5 s as the map fills in, and pure pursuit steers toward a point 0.3 m ahead on the path.{legged ? ' For a walking body the commanded speed and turn rate set the stride and turn of each gait cycle; a run also ends if the body falls over.' : ''}</p>
          </Explain>
        </Section>
        <Section title="Drive">
          <div className="row">
            <button className="btn primary" onClick={() => session.startAuto()} disabled={!session.drivable} data-testid="nav-go">{legged ? 'Walk to goal' : 'Drive to goal'}</button>
            <button className="btn" onClick={() => session.stop()}>Stop</button>
            <button className="btn" onClick={() => session.reset()}>Reset</button>
          </div>
          <p className="note muted">Type a goal or double-click the floor to move it.</p>
          <div className="row" key={`goal:${goalAttempt}:${s.goal[0]}:${s.goal[1]}`}>
            <NumberField label="Goal x (m)" value={s.goal[0]} digits={2} step={0.1} onCommit={(v) => !session.setGoal(v, s.goal[1]) && setGoalAttempt((n) => n + 1)} testId="nav-goal-x" />
            <NumberField label="Goal y (m)" value={s.goal[1]} digits={2} step={0.1} onCommit={(v) => !session.setGoal(s.goal[0], v) && setGoalAttempt((n) => n + 1)} testId="nav-goal-y" />
          </div>
          <div className="row">
            <NumberField label={`Goal speed limit (m/s, ${lo} to ${hi})`} value={s.speedLimit} digits={2} step={legged ? 0.01 : 0.05} min={lo} max={hi} onCommit={(v) => session.setSpeedLimit(v)} testId="nav-speed-limit" />
          </div>
          <p className="note muted">Caps the speed when driving to a goal, at once and also during a run; slower or faster in the command bar change it by {legged ? '0.05 m/s (0.02 m/s for a little)' : '0.1 m/s (0.05 m/s for a little)'}. Limits set for or during an AI task last until that task ends. Keyboard driving is not limited by it.{legged ? ' The limit caps the commanded walking speed; the measured speed depends on how the feet grip and can differ from it.' : ''}</p>
          <div className="row" style={{ marginTop: 8 }}>
            <button className="btn" onClick={() => (remoteOn ? session.stop() : session.setRemote(0, 0))} disabled={s.status !== 'ready'} data-testid="nav-remote" aria-pressed={remoteOn}>
              {remoteOn ? 'End remote control' : 'Remote control'}
            </button>
          </div>
          <p className="note muted" data-testid="nav-keys">Keyboard: arrow keys or W A S D. Up or W drives forward, Down or S backward, Left or A and Right or D turn; release to stop. A drive key takes over from Drive to goal. While a tab or slider has focus only W A S D drive; nothing drives while you type in a field.</p>
        </Section>
        <Section title="Evaluate all maps">
          <div className="row">
            <button className="btn" onClick={() => void session.evaluateAll()} disabled={s.status !== 'ready' || ev.running} data-testid="nav-evaluate">Run the 5 test maps</button>
            {ev.running && <button className="btn" onClick={() => session.cancelEvaluation()} data-testid="nav-evaluate-cancel">Cancel</button>}
          </div>
          <p className="note muted">Runs in a Web Worker with the controller inputs and robot selected when it starts{ev.mode ? ` (this run: ${ev.mode}, ${ev.robotName ?? 'the rover'}${ev.walker ? `, ${ev.walker}` : ''})` : ''}; the live view keeps running.</p>
          {ev.running && <p className="note" role="status">Running: {(ev.progress * 100).toFixed(0)} %. {ev.detail}</p>}
          {!ev.running && ev.detail && ev.results.length < TEST_MAPS.length && <p className="note warn" data-testid="nav-evaluate-status">{ev.detail}.</p>}
          {ev.results.length > 0 && (
            <table className="chain" data-testid="nav-results" style={{ marginTop: 6 }}>
              <thead>
                <tr><th style={{ width: 'auto' }}>Map</th><th style={{ width: 'auto' }}>Result</th><th style={{ width: 'auto' }}>Time</th></tr>
              </thead>
              <tbody>
                {ev.results.map((r) => (
                  <tr key={r.mapId}>
                    <td>{TEST_MAPS.find((m) => m.id === r.mapId)!.name}</td>
                    <td>{r.reached ? 'Reached' : `Failed: ${r.failure}`}</td>
                    <td className="mono">{r.timeS.toFixed(1)} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {!ev.running && ev.results.length === TEST_MAPS.length && <p className="note" data-testid="nav-summary">{reachedCount} of {TEST_MAPS.length} maps reached without collision by {ev.robotName ?? 'the rover'} ({ev.results[0]!.mode}).</p>}
        </Section>
      </aside>
      <NavViewport session={session} memory={memory} />
      <aside className="panel right" aria-label="Navigation status">
        <Section title="Status">
          {s.status === 'loading' && <StatusBadge tone="idle">Loading physics engine</StatusBadge>}
          {s.status === 'error' && <p className="note error" role="alert">{s.error}</p>}
          {s.message && <p className={`note ${s.collided || s.fallen || s.unstable ? 'error' : ''}`} role="status" data-testid="nav-message">{s.message}</p>}
          <Readout
            rows={[
              ['Simulated time', `${f(s.simTime)} s`],
              ['Driving', <span data-testid="nav-driving">{s.driving === 'auto' ? `to goal (${s.navStatus})` : s.driving === 'remote' ? 'remote (keyboard)' : 'stopped'}</span>],
              ['Goal', `(${f(s.goal[0])}, ${f(s.goal[1])}) m`],
              ['Position, ground truth', <span data-testid="nav-truth-x">{s.truth ? `(${f(s.truth.x)}, ${f(s.truth.y)}) m` : 'n/a'}</span>],
              ['Position, controller estimate', s.estimate ? `(${f(s.estimate.x)}, ${f(s.estimate.y)}) m` : s.mode === 'ground-truth' ? 'uses ground truth' : 'n/a'],
              ['Speed, ground truth', <span data-testid="nav-speed">{s.truthSpeed === null ? 'n/a' : `${f(s.truthSpeed)} m/s (limit ${f(s.speedLimit)})`}</span>],
              ['Replans', String(s.replans)],
            ]}
          />
        </Section>
        <Section title="Range rays (sensor)">
          <p className="note muted mono" data-testid="nav-ranges">{s.ranges.length ? s.ranges.map((r) => (r < 0 || r > RAY_MAX_RANGE ? 'none' : r.toFixed(2))).join('  ') : 'n/a'}</p>
          <p className="note muted">Nine rays from -80 to 80 deg, range {RAY_MAX_RANGE} m; "none" means no hit within range.</p>
        </Section>
        <Places session={session} memory={memory} />
      </aside>
    </>
  );
}
