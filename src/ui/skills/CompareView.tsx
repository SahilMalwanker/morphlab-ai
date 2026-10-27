import { useEffect, useMemo, useRef, useState } from 'react';
import { RobotView } from '../../render/robotView';
import { Viewport } from '../../render/viewport';
import { StaticPlot, type StaticSeries } from '../plot';
import type { Attempt } from './session';

/** Lateral distance between the two replayed robots, m. */
const OFFSET_Y = 0.45;

function frameAt(a: Attempt, t: number): number {
  const rel = a.t[0]! + t;
  let lo = 0, hi = a.t.length - 1;
  if (rel >= a.t[hi]!) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (a.t[mid]! <= rel) lo = mid;
    else hi = mid;
  }
  return lo;
}

function durationOf(a: Attempt): number {
  return a.t[a.t.length - 1]! - a.t[0]!;
}

/**
 * Replays two recorded attempts side by side with a shared timeline. The
 * robots are drawn from recorded simulator states; nothing is simulated.
 */
export function CompareView({ left, right, onClose }: { left: Attempt; right: Attempt; onClose: () => void }) {
  const host = useRef<HTMLDivElement>(null);
  const duration = Math.max(durationOf(left), durationOf(right));
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(true);
  const state = useRef({ time: 0, playing: true, lastShown: -1 });
  state.current.playing = playing;

  useEffect(() => {
    const vp = new Viewport(host.current!, { cameraPosition: [1.7, 0, 0.75], target: [0, 0, 0.15] });
    const pairs = [left, right].map((a, i) => {
      const view = new RobotView(a.robot);
      view.root.position.y = i === 0 ? -OFFSET_Y : OFFSET_Y;
      vp.scene.add(view.root);
      const links = a.frames.map((f) => Array.from({ length: f.length / 16 - 1 }, (_, k) => f.subarray(16 * (k + 1), 16 * (k + 2))));
      return { a, view, links };
    });
    vp.start((dt) => {
      const st = state.current;
      if (st.playing) {
        st.time = Math.min(duration, st.time + dt);
        if (st.time >= duration) {
          st.playing = false;
          setPlaying(false);
        }
      }
      for (const p of pairs) {
        const k = frameAt(p.a, st.time);
        p.view.applyTransforms(p.a.frames[k]!.subarray(0, 16), p.links[k]!);
      }
      // Publish the time to React at most about 15 times per second.
      if (Math.abs(st.time - st.lastShown) >= 0.066 || (!st.playing && st.time !== st.lastShown)) {
        st.lastShown = st.time;
        setTime(st.time);
      }
    });
    return () => {
      pairs.forEach((p) => p.view.dispose());
      vp.dispose();
    };
  }, [left, right, duration]);

  const same = left.plot.title === right.plot.title;
  const series = useMemo<StaticSeries[]>(() => {
    const rel = (a: Attempt) => a.t.map((t) => t - a.t[0]!);
    const out: StaticSeries[] = [
      { label: 'A', color: '#58a6ff', t: rel(left), y: left.plot.values },
      { label: 'B', color: '#e3b341', t: rel(right), y: right.plot.values },
    ];
    const th = left.plot.threshold;
    if (same && th !== null && th === right.plot.threshold) out.push({ label: `${th} ${left.plot.unit}`, color: '#8592a3', dashed: true, t: [0, duration], y: [th, th] });
    return out;
  }, [left, right, duration, same]);

  const seek = (t: number) => {
    state.current.time = Math.max(0, Math.min(duration, t));
    state.current.lastShown = -1;
    setTime(state.current.time);
  };
  const summary = (a: Attempt) => `${a.label}: ${a.resultText}.`;

  return (
    <section className="viewport compare" aria-label="Attempt comparison" data-testid="compare-view">
      <div className="compare-canvas" ref={host}>
        <div className="viewport-label">Replay of recorded simulator states, display only: nothing is simulated. Left: A, right: B.</div>
      </div>
      <div className="compare-controls">
        <div className="row">
          <button className="btn" type="button" onClick={() => (time >= duration ? (seek(0), setPlaying(true)) : setPlaying(!playing))} data-testid="compare-play">
            {playing ? 'Pause' : time >= duration ? 'Replay' : 'Play'}
          </button>
          <span className="mono grow" data-testid="compare-clock">{time.toFixed(2)} / {duration.toFixed(2)} s</span>
          <button className="btn" type="button" onClick={onClose} data-testid="compare-close">Close comparison</button>
        </div>
        <input type="range" min={0} max={duration} step={0.01} value={time} onChange={(e) => (setPlaying(false), seek(Number(e.target.value)))} aria-label="Replay time" data-testid="compare-time" />
        <p className="note muted" data-testid="compare-left">A: {summary(left)}</p>
        <p className="note muted" data-testid="compare-right">B: {summary(right)}</p>
        <StaticPlot title={same ? left.plot.title : `A: ${left.plot.title}; B: ${right.plot.title}`} unit={left.plot.unit} series={series} height={150} testId="compare-plot" />
      </div>
    </section>
  );
}
