import { useEffect, useRef } from 'react';
import type { Telemetry } from '../core/control/telemetry';

export interface PlotSeries {
  channel: number;
  label: string;
  color: string;
  dashed?: boolean;
}

export interface PlotSpec {
  title: string;
  unit: string;
  series: PlotSeries[];
  /** Optional symmetric limit lines (for example the effort limit). */
  limit?: number;
  scale?: number;
}

/**
 * Lightweight canvas time plot of the last `windowS` seconds of telemetry.
 * Redraws on its own timer and never routes samples through React state.
 */
export function TimePlot({ getTelemetry, spec, windowS = 5, height = 120, testId }: { getTelemetry: () => Telemetry | null; spec: PlotSpec; windowS?: number; height?: number; testId?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const specRef = useRef(spec);
  specRef.current = spec;
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    const draw = () => {
      const s = specRef.current;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#111415';
      ctx.fillRect(0, 0, w, h);
      const tel = getTelemetry();
      const padL = 44, padR = 8, padT = 16, padB = 16;
      const pw = w - padL - padR, ph = h - padT - padB;
      ctx.font = '11px Bahnschrift, "Segoe UI", sans-serif';
      ctx.fillStyle = '#9caaa6';
      ctx.fillText(`${s.title} (${s.unit})`, padL, 11);
      if (!tel || tel.length < 2) return;
      const scale = s.scale ?? 1;
      const tEnd = tel.time(tel.length - 1);
      const tStart = tEnd - windowS;
      let first = 0;
      while (first < tel.length - 1 && tel.time(first) < tStart) first++;
      let lo = Infinity, hi = -Infinity;
      for (const ser of s.series) {
        for (let i = first; i < tel.length; i++) {
          const v = tel.value(i, ser.channel) * scale;
          if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        }
      }
      if (s.limit !== undefined) { lo = Math.min(lo, -s.limit * scale * 1.05); hi = Math.max(hi, s.limit * scale * 1.05); }
      if (!Number.isFinite(lo)) return;
      if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
      const pad = 0.08 * (hi - lo);
      lo -= pad; hi += pad;
      const X = (t: number) => padL + ((t - tStart) / windowS) * pw;
      const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * ph;
      ctx.strokeStyle = '#3a4548';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k <= 4; k++) { const y = padT + (k / 4) * ph; ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); }
      ctx.stroke();
      ctx.fillStyle = '#9caaa6';
      for (let k = 0; k <= 4; k++) {
        const v = hi - (k / 4) * (hi - lo);
        ctx.fillText(v.toFixed(Math.abs(hi - lo) < 2 ? 2 : 1), 2, padT + (k / 4) * ph + 4);
      }
      if (s.limit !== undefined) {
        ctx.strokeStyle = '#ff9b9b';
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        for (const sign of [1, -1]) { const y = Y(sign * s.limit * scale); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); }
        ctx.stroke();
      }
      for (const ser of s.series) {
        ctx.strokeStyle = ser.color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash(ser.dashed ? [5, 4] : []);
        ctx.beginPath();
        let started = false;
        for (let i = first; i < tel.length; i++) {
          const v = tel.value(i, ser.channel) * scale;
          if (!Number.isFinite(v)) continue;
          const x = X(tel.time(i)), y = Y(v);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
      let lx = padL + pw;
      ctx.textAlign = 'right';
      for (const ser of [...s.series].reverse()) {
        ctx.fillStyle = ser.color;
        ctx.fillText(ser.label, lx, 11);
        lx -= ctx.measureText(ser.label).width + 12;
      }
      ctx.textAlign = 'left';
    };
    const id = window.setInterval(draw, 66);
    draw();
    return () => window.clearInterval(id);
  }, [getTelemetry, windowS]);
  return <canvas ref={ref} style={{ width: '100%', height, display: 'block', borderRadius: 4 }} aria-label={`${spec.title} plot`} role="img" data-testid={testId} />;
}

export interface StaticSeries {
  label: string;
  color: string;
  t: ArrayLike<number>;
  y: ArrayLike<number>;
  dashed?: boolean;
}

/**
 * Canvas plot of recorded data with time and value axes. Redraws when the
 * data or the element size changes.
 */
export function StaticPlot({ title, unit, series, limit, height = 220, testId }: { title: string; unit: string; series: StaticSeries[]; limit?: number; height?: number; testId?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#111415';
      ctx.fillRect(0, 0, w, h);
      const padL = 52, padR = 10, padB = 26;
      const pw = w - padL - padR;
      ctx.font = '11px Bahnschrift, "Segoe UI", sans-serif';
      const titleText = `${title} (${unit})`;
      // Legend in series order, right-aligned; a row ends where the next entry would reach the title or the left edge.
      const titleEnd = padL + ctx.measureText(titleText).width + 16;
      const rows: StaticSeries[][] = [[]];
      let used = 0;
      for (const s of series) {
        const lw = ctx.measureText(s.label).width + 14;
        const avail = padL + pw - (rows.length === 1 ? titleEnd : padL);
        if (rows.at(-1)!.length > 0 && used + lw > avail) {
          rows.push([]);
          used = 0;
        }
        rows.at(-1)!.push(s);
        used += lw;
      }
      const legend: { s: StaticSeries; x: number; y: number }[] = [];
      rows.forEach((r, i) => {
        let x = padL + pw;
        for (const s of [...r].reverse()) {
          legend.push({ s, x, y: 13 + 13 * i });
          x -= ctx.measureText(s.label).width + 14;
        }
      });
      const padT = 20 + 13 * (rows.length - 1);
      const ph = h - padT - padB;
      ctx.fillStyle = '#9caaa6';
      ctx.fillText(titleText, padL, 13);
      let t0 = Infinity, t1 = -Infinity, lo = Infinity, hi = -Infinity;
      for (const s of series) {
        for (let i = 0; i < s.t.length; i++) {
          const v = s.y[i]!;
          if (!Number.isFinite(v)) continue;
          t0 = Math.min(t0, s.t[i]!); t1 = Math.max(t1, s.t[i]!);
          lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
      }
      if (!Number.isFinite(lo) || !(t1 > t0)) {
        ctx.fillText('No data yet.', padL, padT + ph / 2);
        return;
      }
      if (limit !== undefined) { lo = Math.min(lo, -limit * 1.05); hi = Math.max(hi, limit * 1.05); }
      if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
      const pad = 0.06 * (hi - lo);
      lo -= pad; hi += pad;
      const X = (t: number) => padL + ((t - t0) / (t1 - t0)) * pw;
      const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * ph;
      ctx.strokeStyle = '#3a4548';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k <= 4; k++) { const y = padT + (k / 4) * ph; ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); }
      for (let k = 0; k <= 5; k++) { const x = padL + (k / 5) * pw; ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); }
      ctx.stroke();
      ctx.fillStyle = '#9caaa6';
      const digits = Math.abs(hi - lo) < 2 ? 3 : 1;
      for (let k = 0; k <= 4; k++) ctx.fillText((hi - (k / 4) * (hi - lo)).toFixed(digits), 2, padT + (k / 4) * ph + 4);
      ctx.textAlign = 'center';
      for (let k = 0; k <= 5; k++) ctx.fillText(`${(t0 + (k / 5) * (t1 - t0)).toFixed(2)} s`, padL + (k / 5) * pw, h - 8);
      ctx.textAlign = 'left';
      if (limit !== undefined) {
        ctx.strokeStyle = '#ff9b9b';
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        for (const sign of [1, -1]) { const y = Y(sign * limit); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); }
        ctx.stroke();
      }
      for (const s of series) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.6;
        ctx.setLineDash(s.dashed ? [5, 4] : []);
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < s.t.length; i++) {
          const v = s.y[i]!;
          if (!Number.isFinite(v)) continue;
          if (!started) { ctx.moveTo(X(s.t[i]!), Y(v)); started = true; } else ctx.lineTo(X(s.t[i]!), Y(v));
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.textAlign = 'right';
      for (const e of legend) {
        ctx.fillStyle = e.s.color;
        ctx.fillText(e.s.label, e.x, e.y);
      }
      ctx.textAlign = 'left';
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [title, unit, series, limit]);
  return <canvas ref={ref} style={{ width: '100%', height, display: 'block', borderRadius: 4 }} aria-label={`${title} plot`} role="img" data-testid={testId} />;
}
