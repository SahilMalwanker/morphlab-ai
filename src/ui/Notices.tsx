import { useMemo } from 'react';
import { checkCapabilities } from './capabilities';
import type { AppStorage } from './storage';
import { useStore } from './store';

/** Storage problems, unreadable records and missing browser features. Renders nothing when all is well. */
export function Notices({ storage }: { storage: AppStorage }) {
  const s = useStore(storage.store);
  const missing = useMemo(() => checkCapabilities().filter((c) => !c.ok), []);
  if (s.status !== 'unavailable' && s.corrupted.length === 0 && s.notices.length === 0 && missing.length === 0) return null;
  return (
    <div className="notices" role="region" aria-label="Notices">
      {missing.map((c) => (
        <p key={c.name} className="note warn" style={{ margin: 0 }}>{c.name} is not available in this browser: {c.consequence}.</p>
      ))}
      {s.status === 'unavailable' && <p className="note warn" style={{ margin: 0 }} data-testid="storage-unavailable">Local storage is unavailable. {s.error}</p>}
      {s.notices.length > 0 && (
        <div data-testid="restore-notices">
          {s.notices.map((n) => (
            <p key={n} className="note warn" style={{ margin: 0 }}>{n}</p>
          ))}
          <button className="btn" type="button" onClick={() => storage.dismissNotices()}>Dismiss</button>
        </div>
      )}
      {s.corrupted.length > 0 && (
        <div data-testid="corrupted-records">
          <p className="note warn" style={{ margin: 0 }}>
            {s.corrupted.length} stored record{s.corrupted.length > 1 ? 's' : ''} could not be read. They are kept until you remove them.
          </p>
          <ul style={{ margin: '4px 0 0', paddingLeft: 18, fontSize: 12 }}>
            {s.corrupted.map((c) => (
              <li key={`${c.store}:${c.key}`}>
                <span className="mono">{c.store === 'robots' ? 'Robot' : c.store === 'skills' ? 'Skill' : c.store === 'memories' ? 'Memory' : 'Setting'} {c.key}</span>: {c.reasons.slice(0, 2).join(' ')}{' '}
                <button className="btn" type="button" onClick={() => void storage.removeCorrupted(c)}>Remove</button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
