import type { MainModule } from '@mujoco/mujoco';
import type { PdGains } from '../core/control/pd';
import type { Vec3 } from '../core/math/vec3';
import type { RobotDescription } from '../core/model/types';
import { RunningStats, type RunningStatsParts } from '../learning/ars';
import type { PolicyJson } from './controllers';
import type { TrialConditions } from './evaluation';
import type { GaitParams } from './gait';
import { GAIT_POLICY_INPUTS } from './policyGait';
import type { SkillGoal } from './taught';
import { gaitPolicyRollout, type GaitRollout } from './train';
import { posePolicyRollout, type PoseOutcome, type PoseRolloutJob, type PoseRunner } from './trainPose';
import { WALK_POLICY_INPUTS } from './walkPolicy';
import type { SkillWorkerClient } from './workerClient';

/** What every rollout of a batch shares: the body, its stance and the joint gains. */
export interface RolloutSetup {
  desc: RobotDescription;
  stanceQ: number[];
  gains?: PdGains;
}

/** One randomized gait run, as sent to a worker (plain data). */
export interface RolloutJob {
  gait: GaitParams;
  policy: PolicyJson | null;
  conditions: TrialConditions;
  noiseSeed: number;
  target?: Vec3;
  goals?: SkillGoal[];
  training?: boolean;
  trackPose?: boolean;
  /** Walk this command with the walking controller (D29): forward m/s, left m/s, turn rad/s. */
  walkCommand?: Vec3;
  /** Return the policy's input statistics of this run. */
  collectStats?: boolean;
}

export interface RolloutOutcome {
  rollout: GaitRollout;
  stats: RunningStatsParts | null;
}

/** Runs batches of rollouts, in this thread or across workers; the outcomes keep the order of the jobs. */
export interface RolloutRunner {
  readonly workers: number;
  run(setup: RolloutSetup, jobs: RolloutJob[]): Promise<RolloutOutcome[]>;
}

/** A runner whose batches can be stopped: pending runs reject with "Cancelled.". */
export interface CancellableRunner extends RolloutRunner, PoseRunner {
  cancel(): void;
  dispose(): void;
}

export function runRolloutJob(mj: MainModule, setup: RolloutSetup, job: RolloutJob): RolloutOutcome {
  const stats = job.collectStats ? new RunningStats(job.walkCommand ? WALK_POLICY_INPUTS : GAIT_POLICY_INPUTS) : undefined;
  const rollout = gaitPolicyRollout(mj, {
    desc: setup.desc, stanceQ: setup.stanceQ, gait: job.gait, policy: job.policy, conditions: job.conditions, noiseSeed: job.noiseSeed,
    ...(setup.gains ? { gains: setup.gains } : {}), ...(job.target ? { target: job.target } : {}), ...(job.goals ? { goals: job.goals } : {}),
    ...(job.training ? { training: true } : {}), ...(job.trackPose ? { trackPose: true } : {}), ...(job.walkCommand ? { walkCommand: job.walkCommand } : {}), ...(stats ? { stats } : {}),
  });
  return { rollout, stats: stats ? stats.toParts() : null };
}

/** Rollouts one after another in this thread (tests, and measurements without workers). */
export function inProcessRunner(mj: MainModule): RolloutRunner & PoseRunner {
  return {
    workers: 1,
    run: async (setup, jobs) => jobs.map((j) => runRolloutJob(mj, setup, j)),
    runPose: async (setup, jobs) => jobs.map((j) => posePolicyRollout(mj, setup, j)),
  };
}

/** Workers for batch training: one fewer than the logical processors, at least 1 and at most 4. */
export function defaultWorkerCount(hardwareConcurrency: number | undefined): number {
  return Math.max(1, Math.min(4, (hardwareConcurrency ?? 2) - 1));
}

/**
 * A pool of skill workers, each with its own MuJoCo instance. A batch is split
 * into contiguous parts, one per worker, and the outcomes are joined in job
 * order, so results do not depend on the number of workers. Cancelling
 * terminates every worker; the next batch starts fresh ones.
 */
export class RolloutPool implements CancellableRunner {
  private clients: SkillWorkerClient[] = [];

  constructor(private readonly create: () => SkillWorkerClient, readonly workers: number) {
    if (!Number.isInteger(workers) || workers < 1) throw new RangeError('A pool needs at least one worker');
  }

  async run(setup: RolloutSetup, jobs: RolloutJob[]): Promise<RolloutOutcome[]> {
    return this.dispatch<RolloutJob, RolloutOutcome>('rollouts', setup, jobs);
  }

  async runPose(setup: RolloutSetup, jobs: PoseRolloutJob[]): Promise<PoseOutcome[]> {
    return this.dispatch<PoseRolloutJob, PoseOutcome>('pose-rollouts', setup, jobs);
  }

  private async dispatch<J, O>(type: 'rollouts' | 'pose-rollouts', setup: RolloutSetup, jobs: J[]): Promise<O[]> {
    const n = Math.min(this.workers, jobs.length);
    while (this.clients.length < n) this.clients.push(this.create());
    const size = Math.ceil(jobs.length / Math.max(1, n));
    const parts = await Promise.all(
      Array.from({ length: n }, (_, i) => {
        const part = jobs.slice(i * size, (i + 1) * size);
        const common = { robot: setup.desc, limbIndex: -1, stanceQ: setup.stanceQ, ...(setup.gains ? { gains: setup.gains } : {}) };
        return type === 'rollouts'
          ? this.clients[i]!.run<O[]>({ type, ...common, jobs: part as unknown as RolloutJob[] })
          : this.clients[i]!.run<O[]>({ type, ...common, jobs: part as unknown as PoseRolloutJob[] });
      }),
    );
    return parts.flatMap((p) => p.result);
  }

  cancel(): void {
    for (const c of this.clients) c.cancel();
  }

  dispose(): void {
    for (const c of this.clients) c.dispose();
    this.clients = [];
  }
}
