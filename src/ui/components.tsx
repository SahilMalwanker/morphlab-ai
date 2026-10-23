import { useEffect, useId, useRef, useState, type ReactNode } from 'react';

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="section" aria-label={title}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function Toggle({ label, checked, onChange, disabled }: { label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

export function Readout({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="readout">
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: 'contents' }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Numeric input that only commits finite values (on Enter or blur). Invalid
 * text is flagged and never reaches the model.
 */
export function NumberField({
  label,
  value,
  onCommit,
  step = 0.01,
  digits = 3,
  min,
  max,
  disabled,
  testId,
}: {
  label: string;
  value: number;
  onCommit: (v: number) => void;
  step?: number;
  digits?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  testId?: string;
}) {
  const id = useId();
  const [text, setText] = useState(value.toFixed(digits));
  const [invalid, setInvalid] = useState(false);
  useEffect(() => {
    setText(value.toFixed(digits));
    setInvalid(false);
  }, [value, digits]);
  const commit = () => {
    // Leaving a field without editing it must not commit the rounded display value.
    if (text === value.toFixed(digits)) {
      setInvalid(false);
      return;
    }
    const v = Number(text);
    const ok = text.trim() !== '' && Number.isFinite(v) && (min === undefined || v >= min) && (max === undefined || v <= max);
    setInvalid(!ok);
    if (ok) onCommit(v);
  };
  return (
    <div className="field grow">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="number"
        step={step}
        value={text}
        disabled={disabled}
        aria-invalid={invalid}
        data-testid={testId}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit();
        }}
      />
    </div>
  );
}

export function StatusBadge({ tone, children, testId }: { tone: 'ok' | 'warn' | 'error' | 'idle'; children: ReactNode; testId?: string }) {
  return (
    <span className={`status ${tone}`} data-testid={testId} role="status">
      {children}
    </span>
  );
}

export const RAD2DEG = 180 / Math.PI;
export const DEG2RAD = Math.PI / 180;

export function fmt(v: number, digits = 3): string {
  return Number.isFinite(v) ? v.toFixed(digits) : 'n/a';
}

/** Saves text as a JSON file through the browser's download. */
export function downloadText(text: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * A focusable button that opens the file picker for one JSON file and hands
 * over its text. Files larger than `maxBytes` are refused before reading.
 */
export function FileButton({ label, maxBytes, onText, onRefuse, testId }: { label: string; maxBytes: number; onText: (text: string) => void; onRefuse: (message: string) => void; testId?: string }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <button type="button" className="btn" onClick={() => input.current?.click()}>{label}</button>
      <input
        ref={input}
        type="file"
        accept=".json,application/json"
        hidden
        tabIndex={-1}
        aria-hidden="true"
        data-testid={testId}
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = '';
          if (!f) return;
          if (f.size > maxBytes) {
            onRefuse(`The file is larger than ${maxBytes / 1_000_000} MB.`);
            return;
          }
          onText(await f.text());
        }}
      />
    </>
  );
}
