import { useId, useRef, type KeyboardEvent } from 'react';
import { LAB_TOOLS, modeInfo, PLAYGROUND, type Mode } from './modes';

interface WorkspaceNavProps {
  mode: Mode;
  onOpen: (mode: Mode) => void;
  menuOpen: boolean;
  onMenu: (open: boolean) => void;
  guideShown: boolean;
  onGuide: () => void;
  learnMode: boolean;
  onLearn: () => void;
}

/**
 * Primary navigation: the AI Playground first, the engineering tools as a
 * secondary group. On narrow windows the list folds behind a Menu button;
 * that is a disclosure, not a modal dialog, so focus is never trapped.
 */
export function WorkspaceNav({ mode, onOpen, menuOpen, onMenu, guideShown, onGuide, learnMode, onLearn }: WorkspaceNavProps) {
  const bodyId = useId();
  const labId = useId();
  const subId = useId();
  const menuRef = useRef<HTMLButtonElement>(null);
  const close = () => {
    onMenu(false);
    // The list folds away on narrow windows; keep keyboard focus on a visible control.
    menuRef.current?.focus();
  };
  const select = (m: Mode) => {
    onOpen(m);
    if (menuOpen) close();
  };
  const onKey = (e: KeyboardEvent<HTMLElement>) => {
    if (e.key === 'Escape' && menuOpen) {
      e.preventDefault();
      close();
    }
  };
  const item = (m: Mode) => {
    const primary = m === PLAYGROUND;
    const title = modeInfo(m).title;
    return (
      <li key={m}>
        <button
          type="button"
          className={`nav-item${primary ? ' nav-primary' : ''}`}
          aria-current={mode === m ? 'page' : undefined}
          aria-label={primary ? title : undefined}
          aria-describedby={primary ? subId : undefined}
          onClick={() => select(m)}
          data-testid={`nav-${m}`}
        >
          {title}
          {primary && (
            <span id={subId} className="nav-sub">
              Skills: teach, evaluate, run
            </span>
          )}
        </button>
      </li>
    );
  };
  return (
    <nav className="sidenav" aria-label="Workspace" onKeyDown={onKey}>
      <div className="sidenav-top">
        <h1 className="brand">MorphLab</h1>
        <span className="sidenav-current">{modeInfo(mode).title}</span>
        <button ref={menuRef} type="button" className="btn nav-menu" aria-expanded={menuOpen} aria-controls={bodyId} onClick={() => onMenu(!menuOpen)} data-testid="nav-menu">
          Menu
        </button>
      </div>
      <div id={bodyId} className={`sidenav-body${menuOpen ? ' open' : ''}`}>
        <ul className="nav-list">{item(PLAYGROUND)}</ul>
        <h2 className="nav-group" id={labId}>Lab tools</h2>
        <ul className="nav-list" aria-labelledby={labId}>
          {LAB_TOOLS.map((m) => item(m))}
        </ul>
        <div className="nav-utility">
          <button className="btn" type="button" aria-pressed={guideShown} onClick={onGuide} data-testid="guide-toggle">
            Guide
          </button>
          <button className="btn" type="button" aria-pressed={learnMode} onClick={onLearn} data-testid="learn-toggle">
            Learn mode
          </button>
          <p className="nav-meta">SI units, radians internally, world frame Z up</p>
        </div>
      </div>
    </nav>
  );
}
