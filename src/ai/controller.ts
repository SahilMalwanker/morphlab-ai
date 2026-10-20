import type { RobotCapabilities } from './capabilities.ts';
import type { AiAvailability } from './client.ts';
import type { ToolStep, TurnReport } from './loop.ts';
import type { ChatMessage, ChatRequest, ChatResult } from './protocol.ts';
import { rejected, type ToolResult } from './result.ts';
import type { TaskChange, TaskExecutor, TaskManager } from './tasks.ts';

const MAX_MESSAGE_CHARS = 16_000;

/** What the active view offers the AI: capabilities, read-only queries and task executors. */
export interface BridgeSnapshot {
  caps: RobotCapabilities;
  /** One paragraph about the robot and scene for the system prompt. */
  summary: string;
  /** Read-only tools (state, capabilities, skill list); returns measured data. */
  query(tool: string, args: Record<string, unknown>): ToolResult;
  /** Action tools: an executor for the task manager, or an immediate result (memory changes and refusals). `ctx.request` is the user's command. */
  action(tool: string, args: Record<string, unknown>, ctx?: { request: string }): TaskExecutor | ToolResult;
  /** A reminder when the model answers this view's request without acting on it (see TurnOptions.followUp); null accepts the answer. */
  followUp?(steps: readonly ToolStep[], request: string): string | null;
}

export interface RobotBridge {
  /** Changes whenever the view, robot or scene changes; plans made under another key are stale. */
  contextKey(): string;
  snapshot(): Promise<BridgeSnapshot | { unsupported: string }>;
}

export interface AiClientLike {
  status(): Promise<AiAvailability>;
  chat(request: ChatRequest, signal: AbortSignal): Promise<ChatResult>;
}

export type AiCommandResult =
  | { kind: 'unavailable'; reason: string }
  | { kind: 'unsupported'; reason: string }
  | { kind: 'busy' }
  | { kind: 'turn'; report: TurnReport };

const STATUS_TTL_MS = 30_000;
/** Tools that change the robot in view on purpose. */
const CONTEXT_TOOLS: ReadonlySet<string> = new Set(['selectRobot']);
// Four exchanges of three messages: the request, the reply and a note on the tool results behind it.
const HISTORY_MESSAGES = 12;
const HISTORY_CHARS = 1000;

export function systemPrompt(summary: string): string {
  const text = [
    'You operate a simulated robot in MorphLab, a robotics learning playground, only through the tools provided.',
    'Every tool result is measured in the simulator. Report outcomes only from tool results: never say an action succeeded unless its result has status "succeeded"; for "failed", "cancelled" or "rejected", say so and give the reason and the fallback.',
    'Call at most one action tool per answer and wait for its result.',
    'You never send torques, motor commands or joint targets to the robot while it moves; refuse such requests and say which tools exist instead. Poses in a skill draft are not commands: the draft is tried in the simulator and judged by its goals.',
    'Remembered places and skill run records were stored in this browser from earlier measurements and requests; they are records, not current measurements. Call a tool instead of asking the user for something a tool measures or looks up itself. Questions about the robot\'s current position, state, distances or surroundings need a fresh tool result: call the matching tool first and answer from it, not from this description or earlier replies.',
    'Units are SI: metres, seconds, radians. The world frame is right-handed with z up.',
    'Answer in one to three short sentences of plain text: no markdown, and symbols as plain words (kd, not \\(kd\\)).',
    summary,
  ].join('\n');
  // The proxy refuses longer messages; summaries put what may be cut (long lists) last.
  const cut = ' [cut to fit the request limit]';
  return text.length <= MAX_MESSAGE_CHARS ? text : text.slice(0, MAX_MESSAGE_CHARS - cut.length) + cut;
}

function isExecutor(x: TaskExecutor | ToolResult): x is TaskExecutor {
  return typeof (x as TaskExecutor).run === 'function';
}

/**
 * Kept after each remembered reply. Without it the model saw earlier answers
 * quote values with no visible source and imitated them: it invented sensor
 * readings and claimed results without calling a tool. A system message,
 * unlike a tag in the reply, is not copied into new answers.
 */
function evidenceNote(report: TurnReport): ChatMessage {
  const used = report.steps.filter((s) => s.accepted).map((s) => `${s.tool} ${s.result.status}`);
  return {
    role: 'system',
    content: used.length
      ? `Note: the reply above was written from tool results measured then (${used.join(', ')}); it is not a current measurement.`
      : 'Note: the reply above used no tool result; it contains no measurement.',
  };
}

/** Runs AI commands through the proxy with validated tools, one turn at a time. */
export class AiController {
  readonly tasks: TaskManager;
  readonly #client: AiClientLike;
  readonly #now: () => number;
  #status: AiAvailability = { state: 'unknown' };
  #statusAt = Number.NEGATIVE_INFINITY;
  #turn: AbortController | null = null;
  #history: ChatMessage[] = [];
  #historyKey: string | null = null;

  constructor(client: AiClientLike, tasks: TaskManager, now: () => number = Date.now) {
    this.#client = client;
    this.tasks = tasks;
    this.#now = now;
  }

  get status(): AiAvailability {
    return this.#status;
  }

  get busy(): boolean {
    return this.#turn !== null;
  }

  /** Conversation memory for the current robot and scene (cleared when they change). */
  get history(): readonly ChatMessage[] {
    return this.#history;
  }

  async refreshStatus(force = false): Promise<AiAvailability> {
    if (!force && this.#status.state !== 'unknown' && this.#now() - this.#statusAt < STATUS_TTL_MS) return this.#status;
    this.#status = await this.#client.status();
    this.#statusAt = this.#now();
    return this.#status;
  }

  async run(text: string, bridge: RobotBridge, onStep?: (step: ToolStep) => void): Promise<AiCommandResult> {
    if (this.#turn) return { kind: 'busy' };
    const status = await this.refreshStatus();
    if (status.state !== 'available') return { kind: 'unavailable', reason: status.state === 'unavailable' ? status.reason : 'The AI proxy has not been checked yet.' };
    const snap0 = await bridge.snapshot();
    if ('unsupported' in snap0) return { kind: 'unsupported', reason: snap0.unsupported };
    const [{ runAiTurn, skillFollowUp }, { toolsForModel }] = await Promise.all([import('./loop.ts'), import('./tools.ts')]);
    let snap = snap0;
    let key = bridge.contextKey();
    if (key !== this.#historyKey) {
      this.#history = [];
      this.#historyKey = key;
    }
    let planned = this.tasks.stateVersion;
    const turn = new AbortController();
    this.#turn = turn;
    const execute = async (name: string, args: Record<string, unknown>): Promise<ToolResult> => {
      if (bridge.contextKey() !== key) this.tasks.invalidate('the robot or scene changed during the conversation');
      if (name === 'cancelTask') return this.tasks.cancel(String(args['taskId']), planned);
      if (name === 'modifyTask') return this.tasks.modify(String(args['taskId']), planned, { change: args['change'], amount: args['amount'] } as TaskChange);
      if (this.tasks.stateVersion !== planned) return rejected('The robot or scene changed after this command was planned; ask again.');
      if (name === 'getRobotCapabilities' || name === 'getRobotState' || name === 'listSkills' || name === 'getSceneObjects') {
        return snap.query(name, args);
      }
      const x = snap.action(name, args, { request: text });
      const result = isExecutor(x) ? await this.tasks.start(name, planned, x).done : x;
      if (CONTEXT_TOOLS.has(name) && result.status === 'succeeded' && bridge.contextKey() !== key) {
        // The command itself changed the robot, so what it plans next is planned against the new one.
        this.tasks.invalidate('the command switched the robot');
        key = bridge.contextKey();
        planned = this.tasks.stateVersion;
      }
      return result;
    };
    // After an action the view may offer more (a skill saved during this command, another robot it switched to).
    const refresh = async (): Promise<{ caps: RobotCapabilities; tools: ReturnType<typeof toolsForModel>; system: string } | null> => {
      if (bridge.contextKey() !== key) return null;
      const next = await bridge.snapshot();
      if ('unsupported' in next) return null;
      snap = next;
      return { caps: next.caps, tools: toolsForModel(next.caps), system: systemPrompt(next.summary) };
    };
    try {
      const report = await runAiTurn({
        chat: (r, s) => this.#client.chat(r, s),
        system: systemPrompt(snap.summary),
        history: this.#history,
        userText: text,
        tools: toolsForModel(snap.caps),
        caps: snap.caps,
        execute,
        refresh,
        followUp: (steps) => skillFollowUp(steps, text) ?? snap.followUp?.(steps, text) ?? null,
        signal: turn.signal,
        ...(onStep ? { onStep } : {}),
      });
      if (report.end === 'error' && report.error) {
        // A failed request may mean the proxy went away; check again next time.
        this.#statusAt = Number.NEGATIVE_INFINITY;
      }
      if (report.end === 'answered' && report.text) {
        this.#history = [...this.#history, { role: 'user' as const, content: text.slice(0, HISTORY_CHARS) }, { role: 'assistant' as const, content: report.text.slice(0, HISTORY_CHARS) }, evidenceNote(report)].slice(-HISTORY_MESSAGES);
      }
      return { kind: 'turn', report };
    } finally {
      this.#turn = null;
    }
  }

  /** The Stop control: ends the AI turn and applies the safe stop, with or without AI. */
  stop(): ToolResult | null {
    this.#turn?.abort();
    return this.tasks.stopAll();
  }

  forget(): void {
    this.#history = [];
  }
}
