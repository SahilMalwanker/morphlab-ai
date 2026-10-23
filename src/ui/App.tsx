import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { ProxyClient } from '../ai/client';
import { AiController, type RobotBridge } from '../ai/controller';
import type { ToolStep } from '../ai/loop';
import { LongTermMemory } from '../ai/memory';
import { TaskManager } from '../ai/tasks';
import { arm3, quadruped } from '../core/model/presets';
import type { RobotDescription } from '../core/model/types';
import { CommandInterpreter } from '../language/interpreter';
import { parseCommand } from '../language/parser';
import { loadMujocoBrowser } from '../physics/loader';
import { SkillWorkerClient } from '../skills/workerClient';
import type { WorkspaceAccess } from './ai/bridge';
import { applyLiveCommand } from './ai/live';
import type { BuilderSession } from './builder/session';
import { CommandBar } from './CommandBar';
import type { ControlSession } from './control/session';
import { ErrorBoundary } from './ErrorBoundary';
import { Guide } from './Guide';
import type { KinematicsSession } from './kinematics/session';
import { LearnContext } from './learn';
import { modeInfo, PLAYGROUND, type Mode } from './modes';
import type { NavSession } from './nav/session';
import { Notices } from './Notices';
import { PhysicsSession } from './physics/session';
import { SkillsMode } from './skills/SkillsMode';
import { SkillsSession, type CommandOutcome, type CommandTarget } from './skills/session';
import { AppStorage, type RestoredWorkspace, type UiPrefs } from './storage';
import { WorkspaceNav } from './WorkspaceNav';

type LabSessions = typeof import('./labSessions');

const BuilderMode = lazy(() => import('./builder/BuilderMode').then((module) => ({ default: module.BuilderMode })));
const KinematicsMode = lazy(() => import('./kinematics/KinematicsMode').then((module) => ({ default: module.KinematicsMode })));
const PhysicsMode = lazy(() => import('./physics/PhysicsMode').then((module) => ({ default: module.PhysicsMode })));
const ControlMode = lazy(() => import('./control/ControlMode').then((module) => ({ default: module.ControlMode })));
const NavMode = lazy(() => import('./nav/NavMode').then((module) => ({ default: module.NavMode })));

/** Opens local storage first so the last workspace can be restored, then shows the app. */
export function App() {
  const storage = useMemo(() => new AppStorage(), []);
  const [restored, setRestored] = useState<RestoredWorkspace | null>(null);
  useEffect(() => {
    let alive = true;
    void storage.open().then((r) => {
      if (alive) setRestored(r);
    });
    return () => {
      alive = false;
    };
  }, [storage]);
  if (!restored) {
    return (
      <div className="app" aria-busy="true">
        <p className="note" role="status" style={{ padding: 24 }}>Opening local storage.</p>
      </div>
    );
  }
  return <Workspace storage={storage} restored={restored} />;
}

function Workspace({ storage, restored }: { storage: AppStorage; restored: RestoredWorkspace }) {
  const [mode, setMode] = useState<Mode>(PLAYGROUND);
  const [prefs, setPrefs] = useState<UiPrefs>(restored.prefs);
  const [menuOpen, setMenuOpen] = useState(false);
  // The Playground side panel renders into this element, after the command bar (see theme.css, .workspace.pg).
  const [sideSlot, setSideSlot] = useState<HTMLDivElement | null>(null);
  const mainRef = useRef<HTMLElement>(null);
  const shownMode = useRef(mode);
  const updatePrefs = (patch: Partial<UiPrefs>) => {
    const next = { ...prefs, ...patch };
    setPrefs(next);
    storage.savePrefs(next).catch((e: unknown) => console.warn('Preferences were not saved', e));
  };
  const interpreter = useMemo(() => new CommandInterpreter(), []);
  const [lab, setLab] = useState<LabSessions | null>(null);
  const labRef = useRef<LabSessions | null>(null);
  const builderRef = useRef<BuilderSession | null>(null);
  const kinematicsRef = useRef<KinematicsSession | null>(null);
  const physicsRef = useRef<PhysicsSession | null>(null);
  const controlRef = useRef<ControlSession | null>(null);
  const skillsRef = useRef<SkillsSession | null>(null);
  const navRef = useRef<NavSession | null>(null);
  const modeRef = useRef<Mode>(mode);
  modeRef.current = mode;
  const skillsFn = useRef<(() => SkillsSession) | null>(null);
  const ensureLab = async (): Promise<LabSessions> => {
    if (labRef.current) return labRef.current;
    const loaded = await import('./labSessions');
    labRef.current = loaded;
    setLab(loaded);
    return loaded;
  };
  // The Stop control and AI tasks act on the body of the tab they started in; no AI service is needed to stop.
  const tasks = useMemo(
    () =>
      new TaskManager({
        body: () =>
          modeRef.current === 'navigate'
            ? navRef.current?.drive.kind === 'legged'
              ? 'legged'
              : 'wheeled'
            : modeRef.current === 'skills'
              ? skillsRef.current?.state.robot.base.kind === 'fixed'
                ? 'arm'
                : 'legged'
              : 'arm',
        stop: (behavior) => {
          const nav = navRef.current;
          if (behavior === 'brake') {
            if (nav?.drive.kind === 'wheeled' && nav.state.driving !== 'idle') nav.stop('Stopped with the Stop control.');
          } else if (behavior === 'stance') {
            // A legged robot walking in Navigate stops stepping and stands; the Skills view returns to its stance.
            if (nav?.drive.kind === 'legged' && nav.state.driving !== 'idle') nav.stop('Stopped with the Stop control.');
            skillsRef.current?.stopActivity('the Stop control was pressed');
          } else if (behavior === 'hold' && skillsRef.current?.state.robot.base.kind === 'fixed') {
            skillsRef.current?.stopActivity('the Stop control was pressed');
          }
        },
      }),
    [],
  );
  const ai = useMemo(() => new AiController(new ProxyClient(), tasks), [tasks]);
  const ready = storage.state.status === 'ready';
  // World and skill memory, restored from storage; without storage it lasts until the page closes.
  const memory = useMemo(() => {
    const m = new LongTermMemory(restored.memories);
    if (ready) m.attach(storage.memoryHooks());
    return m;
  }, []);
  const bridge = useMemo<RobotBridge>(
    () => {
      const access: WorkspaceAccess = {
        mode: () => modeRef.current,
        nav: () => navRef.current,
        skills: () => skillsFn.current!(),
        memory: () => memory,
        control: () => controlRef.current,
        physics: () => physicsRef.current,
        kinematics: () => kinematicsRef.current,
        builder: () => builderRef.current,
        builtRobot: () => builderRef.current?.state.robot ?? restored.builderRobot ?? null,
      };
      let loaded: RobotBridge | null = null;
      return {
        contextKey: () => loaded?.contextKey() ?? modeRef.current,
        snapshot: async () => {
          loaded ??= (await import('./ai/bridge')).workspaceBridge(access);
          return loaded.snapshot();
        },
      };
    },
    [memory],
  );
  useEffect(() => {
    tasks.invalidate('the view changed');
    if (mode !== 'skills' && mode !== 'physics' && !labRef.current) void ensureLab();
  }, [mode, tasks]);
  // A view switch can remove the focused control (Back to AI Playground, the links in Build);
  // focus then moves to the new view instead of falling back to the page body.
  useEffect(() => {
    if (shownMode.current === mode) return;
    shownMode.current = mode;
    if (document.activeElement === null || document.activeElement === document.body) mainRef.current?.focus();
  }, [mode]);
  const builder = (l: LabSessions): BuilderSession => {
    if (!builderRef.current) {
      builderRef.current = new l.BuilderSession(restored.builderRobot ?? quadruped());
      if (ready) builderRef.current.attachPersistence(storage.builderHooks());
    }
    return builderRef.current;
  };
  const kinematics = (l: LabSessions): KinematicsSession => (kinematicsRef.current ??= new l.KinematicsSession(arm3()));
  const physics = (): PhysicsSession => (physicsRef.current ??= new PhysicsSession(loadMujocoBrowser));
  const control = (l: LabSessions): ControlSession => (controlRef.current ??= new l.ControlSession(loadMujocoBrowser));
  const skills = (): SkillsSession => {
    if (!skillsRef.current) {
      const s = new SkillsSession(new PhysicsSession(loadMujocoBrowser), new SkillWorkerClient(), restored.skillsRobot ?? quadruped(), interpreter);
      s.addSkills(restored.skills);
      s.addLocomotion(restored.locomotion);
      if (ready) s.attachPersistence(storage.skillsHooks());
      s.attachMemory(memory);
      skillsRef.current = s;
    }
    return skillsRef.current;
  };
  skillsFn.current = skills;
  const navigate = (l: LabSessions): NavSession => {
    if (!navRef.current) {
      const n = new l.NavSession(loadMujocoBrowser);
      // Walking controllers are trained in the AI Playground's workers and stored per morphology for every view.
      n.attachWalking({
        controllerFor: (hash) => (skillsRef.current ? skillsRef.current.walkingController(hash) : (restored.locomotion.find((c) => c.robot.morphologyHash === hash) ?? null)),
        train: (robot, onProgress, signal) => skills().trainWalkingFor(robot, onProgress, signal),
      });
      navRef.current = n;
    }
    return navRef.current;
  };
  if (mode === 'build' && lab) builder(lab);
  if (mode === 'kinematics' && lab) kinematics(lab);
  if (mode === 'physics') physics();
  if (mode === 'control' && lab) control(lab);
  if (mode === 'skills') skills();
  if (mode === 'navigate' && lab) navigate(lab);
  const openKinematics = (d: RobotDescription) => {
    if (labRef.current) kinematics(labRef.current).setRobot(d);
    setMode('kinematics');
  };
  const simulate = (d: RobotDescription) => {
    physics().loadRobot(d);
    setMode('physics');
  };
  const openControl = (d: RobotDescription) => {
    if (labRef.current) control(labRef.current).setRobot(d);
    setMode('control');
  };
  const openSkills = (d: RobotDescription) => {
    skills().setRobot(d);
    setMode('skills');
  };
  /** A refused body leaves the current robot in place; Navigate shows why. */
  const openNavigate = (d: RobotDescription) => {
    if (labRef.current) navigate(labRef.current).setRobot(d, 'custom');
    setMode('navigate');
  };
  /** What a command typed outside the Skills tab may act on: that tab's robot and live simulation. */
  const commandTarget = (l: LabSessions | null): CommandTarget => {
    const tab = modeInfo(mode).label;
    const showSkills = () => setMode('skills');
    switch (mode) {
      case 'physics': {
        const p = physics();
        const reset = () => {
          p.reset();
          return 'Physics: reset the simulation; the robot stands in its home pose under the joint controller.';
        };
        return { tab, robot: p.state.description, simulation: p.runtime ? { stand: reset, reset } : null, showSkills };
      }
      case 'navigate': {
        const n = navRef.current ?? (l ? navigate(l) : null);
        const sim = n?.runtime
          ? {
              stand: () => {
                n.stop();
                return `Navigate: ${n.robot.name} stopped.`;
              },
              reset: () => {
                n.reset();
                return `Navigate: ${n.robot.name} is back at the start of the map.`;
              },
            }
          : null;
        return { tab, robot: n?.runtime?.description ?? null, simulation: sim, showSkills };
      }
      case 'build':
        return { tab, robot: l ? builder(l).state.robot : (restored.builderRobot ?? quadruped()), simulation: null, showSkills };
      case 'kinematics':
        return { tab, robot: l ? kinematics(l).state.description : arm3(), simulation: null, showSkills };
      case 'control':
        return { tab, robot: l ? control(l).state.robot : arm3(), simulation: null, showSkills };
      default:
        return { tab, robot: null, simulation: null, showSkills };
    }
  };
  /** Local rules first; requests they do not cover go to the AI path when the proxy is available. */
  const runCommand = async (text: string, onStep: (step: ToolStep) => void): Promise<CommandOutcome> => {
    const live = liveCommand(text);
    if (live !== null) return { command: text, kind: 'done', text: live, source: 'local rules', note: null };
    const loadedLab = mode !== 'skills' && mode !== 'physics' ? await ensureLab() : labRef.current;
    const target = mode === 'skills' ? undefined : commandTarget(loadedLab);
    // A skill the user taught runs by its name, before any other interpretation.
    if (skills().taughtSkillFor(text)) return skills().runCommand(text, target);
    const robot = (target ? target.robot : null) ?? skills().state.robot;
    const parsed = parseCommand(text, { robotName: robot.name, limbs: robot.limbs.map((l) => ({ id: l.id, name: l.name, role: l.role })) });
    if (parsed.kind === 'refusal' && (parsed.escalate || skills().taughtSkillMentioned(text))) {
      const status = await ai.refreshStatus();
      if (status.state === 'available') {
        const { aiOutcome } = await import('./ai/outcome');
        const r = await ai.run(text, bridge, onStep);
        if (r.kind !== 'unsupported') return aiOutcome(text, r);
        // This tab has no AI tools: the local answer says what to do, and the note says where the AI can act.
        const local = await skills().runCommand(text, target);
        return { ...local, note: [local.note, r.reason].filter(Boolean).join(' ') };
      }
    }
    // A lift that no evaluated skill serves goes to the AI in the Skills tab, where it can compose one.
    const escalate =
      mode === 'skills'
        ? async (): Promise<CommandOutcome | null> => {
            if ((await ai.refreshStatus()).state !== 'available') return null;
            const { aiOutcome } = await import('./ai/outcome');
            return aiOutcome(text, await ai.run(text, bridge, onStep));
          }
        : undefined;
    return skills().runCommand(text, target, escalate);
  };
  const stopAll = (): string => {
    const sk = mode === 'skills' ? skillsRef.current?.state : undefined;
    const wasActive = sk ? sk.playing !== null || sk.teleop.active : false;
    const nav = mode === 'navigate' ? navRef.current : null;
    const navWasActive = nav ? nav.state.driving !== 'idle' : false;
    const r = ai.stop();
    if (r) return [r.reason, r.fallback].filter(Boolean).join(' ');
    if (mode === 'navigate') {
      if (!nav) return 'Nothing is running.';
      if (!navWasActive) return 'No navigation run was active. Training and evaluation jobs are not affected; cancel them next to their progress.';
      return nav.drive.kind === 'legged' ? 'The robot stops walking and stands.' : 'The rover brakes to a stop.';
    }
    if (mode === 'skills') {
      return wasActive
        ? sk?.robot.base.kind === 'fixed'
          ? 'Playback and demonstration control stopped; the arm holds its measured pose.'
          : 'Playback and demonstration control stopped; the robot returns to its stance.'
        : 'No playback or demonstration was running. Training and evaluation jobs are not affected; cancel them next to their progress.';
    }
    return 'Nothing in this view runs under command or AI control.';
  };
  /** Slower, faster or stop for what runs now; handled locally, so it works while an AI command waits. */
  const liveCommand = (text: string): string | null => applyLiveCommand(text, { tasks, stop: stopAll, nav: mode === 'navigate' ? navRef.current : null });
  // Dispose every session when the workspace really unmounts. Disposal is deferred one task so
  // StrictMode's immediate unmount and remount in development keeps the sessions alive.
  const disposeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (disposeTimer.current !== null) {
      clearTimeout(disposeTimer.current);
      disposeTimer.current = null;
    }
    return () => {
      disposeTimer.current = setTimeout(() => {
        ai.stop();
        physicsRef.current?.dispose();
        controlRef.current?.dispose();
        skillsRef.current?.dispose();
        navRef.current?.dispose();
        physicsRef.current = controlRef.current = skillsRef.current = navRef.current = null;
      }, 0);
    };
  }, []);
  const title = modeInfo(mode).title;
  const playground = mode === PLAYGROUND;
  const openMode = (m: Mode) => {
    if (m !== 'skills' && m !== 'physics' && !labRef.current) void ensureLab();
    setMode(m);
  };
  return (
    <LearnContext.Provider value={prefs.learnMode}>
      <div className="app">
        <WorkspaceNav
          mode={mode}
          onOpen={openMode}
          menuOpen={menuOpen}
          onMenu={setMenuOpen}
          guideShown={!prefs.guideHidden}
          onGuide={() => updatePrefs({ guideHidden: !prefs.guideHidden })}
          learnMode={prefs.learnMode}
          onLearn={() => updatePrefs({ learnMode: !prefs.learnMode })}
        />
        <div className={`workspace${playground ? ' pg' : ''}${playground && prefs.playgroundPanelCollapsed ? ' side-collapsed' : ''}`}>
          <div className="notice-area">
            {!prefs.guideHidden && <Guide onOpen={openMode} onHide={() => updatePrefs({ guideHidden: true })} />}
            <Notices storage={storage} />
          </div>
          {!playground && (
            <div className="lab-head">
              <h2 className="lab-title">
                <span className="muted">Lab tools / </span>
                {title}
              </h2>
              <button className="btn" type="button" onClick={() => setMode(PLAYGROUND)} data-testid="back-to-playground">
                Back to AI Playground
              </button>
            </div>
          )}
          <main ref={mainRef} tabIndex={-1} className={`workarea mode-${mode}`} aria-label={title} data-testid="workarea">
            <ErrorBoundary key={mode} label={title}>
              <Suspense fallback={<p role="status" className="note">Loading {title}.</p>}>
              {mode === 'build' && builderRef.current && <BuilderMode session={builderRef.current} storage={storage} onOpenKinematics={openKinematics} onSimulate={simulate} onOpenControl={openControl} onOpenSkills={openSkills} onOpenNavigate={openNavigate} />}
              {mode === 'kinematics' && kinematicsRef.current && <KinematicsMode session={kinematicsRef.current} />}
              {mode === 'physics' && physicsRef.current && <PhysicsMode session={physicsRef.current} />}
              {mode === 'control' && controlRef.current && <ControlMode session={controlRef.current} />}
              {mode === 'skills' && skillsRef.current && (
                <SkillsMode session={skillsRef.current} sideSlot={sideSlot} panelCollapsed={prefs.playgroundPanelCollapsed} onPanelCollapsed={(v) => updatePrefs({ playgroundPanelCollapsed: v })} />
              )}
              {mode === 'navigate' && navRef.current && <NavMode session={navRef.current} memory={memory} buildRobot={builderRef.current?.state.robot ?? restored.builderRobot} />}
              </Suspense>
            </ErrorBoundary>
          </main>
          <CommandBar
            ai={ai}
            onRun={runCommand}
            onLive={liveCommand}
            onStop={stopAll}
            hint={modeInfo(mode).hint}
            voice={{ speakReplies: prefs.speakReplies, voiceInputConsent: prefs.voiceInputConsent }}
            onVoice={(patch) => updatePrefs(patch)}
          />
          <div className="pg-side-slot" ref={setSideSlot} />
        </div>
      </div>
    </LearnContext.Provider>
  );
}
