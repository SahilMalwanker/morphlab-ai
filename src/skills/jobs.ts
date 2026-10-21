import type { MainModule } from '@mujoco/mujoco';
import type { PdGains } from '../core/control/pd';
import type { RobotDescription } from '../core/model/types';
import type { SkillSpec } from './controllers';
import { evaluateSkill, recordExpertDemonstration, type EvaluationOptions } from './evaluation';
import type { FootLiftParams } from './footLift';
import type { GaitParams } from './gait';
import { improveTaughtSkill, type TaughtImproveConfig } from './improve';
import { collectDemonstrations, improveParameters, trainImitation, type ImitationConfig, type ImproveConfig } from './learning';
import { runRolloutJob, type RolloutJob } from './rollouts';
import type { SkillGoal } from './taught';
import { trainGaitPolicy, type GaitPolicyConfig } from './train';
import { posePolicyRollout, type PoseRolloutJob } from './trainPose';

interface Common {
  id: number;
  robot: RobotDescription;
  limbIndex: number;
  stanceQ: number[];
}

export type SkillWorkerRequest =
  | (Common & { type: 'evaluate'; spec: SkillSpec; options: EvaluationOptions; goals?: SkillGoal[] })
  | (Common & { type: 'demonstrate'; params: FootLiftParams; gains?: PdGains })
  | (Common & { type: 'imitation'; expert: FootLiftParams; config: ImitationConfig })
  | (Common & { type: 'improve'; base: FootLiftParams; config: ImproveConfig })
  | (Common & { type: 'improve-taught'; spec: SkillSpec; goals: SkillGoal[]; config: TaughtImproveConfig })
  | (Common & { type: 'train-gait-policy'; gait: GaitParams; goals: SkillGoal[]; config: GaitPolicyConfig })
  | (Common & { type: 'rollouts'; gains?: PdGains; jobs: RolloutJob[] })
  | (Common & { type: 'pose-rollouts'; gains?: PdGains; jobs: PoseRolloutJob[] });

export type SkillWorkerMessage =
  | { id: number; type: 'progress'; fraction: number; detail: string }
  | { id: number; type: 'result'; result: unknown; wallTimeS: number }
  | { id: number; type: 'error'; message: string };

/** Runs one skill job with a loaded MuJoCo module, reporting through `post`. Shared by the worker and tests. */
export async function handleSkillJob(loadModule: () => Promise<MainModule>, req: SkillWorkerRequest, post: (m: SkillWorkerMessage) => void): Promise<void> {
  const progress = (fraction: number, detail: string) => post({ id: req.id, type: 'progress', fraction, detail });
  try {
    const t0 = performance.now();
    progress(0, 'Loading the physics engine in the worker');
    const mj = await loadModule();
    let result: unknown;
    switch (req.type) {
      case 'evaluate': {
        const n = req.options.trials;
        let ok = 0;
        result = evaluateSkill(mj, req.robot, req.limbIndex, req.stanceQ, req.spec, req.options, req.goals ? { goals: req.goals } : undefined, (r) => {
          if (r.success) ok++;
          progress((r.trial + 1) / n, `Trial ${r.trial + 1} of ${n}: ${ok} successful so far`);
        });
        break;
      }
      case 'demonstrate':
        result = recordExpertDemonstration(mj, req.robot, req.limbIndex, req.stanceQ, req.params, req.gains);
        break;
      case 'imitation': {
        const cfg = req.config;
        const demos = collectDemonstrations(mj, req.robot, req.limbIndex, req.stanceQ, req.expert, cfg, (ep) => progress(((ep + 1) / cfg.episodes) * 0.3, `Recorded expert episode ${ep + 1} of ${cfg.episodes}`));
        const trained = trainImitation(demos, cfg, (it, loss) => progress(0.3 + 0.7 * ((it + 1) / cfg.iterations), `Training step ${it + 1} of ${cfg.iterations}, loss ${loss.toExponential(2)}`));
        result = { ...trained, horizonS: demos.horizonS };
        break;
      }
      case 'improve': {
        const cfg = req.config;
        result = improveParameters(mj, req.robot, req.limbIndex, req.stanceQ, req.base, cfg, (p) => {
          progress((p.iteration + 1) / cfg.iterations, `Iteration ${p.iteration + 1} of ${cfg.iterations}: mean reward ${p.meanFitness.toFixed(3)}`);
        });
        break;
      }
      case 'improve-taught': {
        const cfg = req.config;
        result = improveTaughtSkill(mj, req.robot, req.stanceQ, req.spec, req.goals, cfg, (p) => {
          progress((p.iteration + 1) / (cfg.iterations + 1), `Iteration ${p.iteration + 1} of ${cfg.iterations}: mean reward ${p.meanFitness.toFixed(3)}, best ${p.bestFitness.toFixed(3)}`);
        });
        break;
      }
      case 'train-gait-policy': {
        const cfg = req.config;
        result = trainGaitPolicy(mj, req.robot, req.stanceQ, req.gait, req.goals, cfg, (p) => {
          progress((p.iteration + 1) / (cfg.iterations + 1), `Iteration ${p.iteration + 1} of ${cfg.iterations}: mean return ${p.meanReturn.toFixed(3)}, best ${p.maxReturn.toFixed(3)} (${p.rollouts} runs)`);
        });
        break;
      }
      case 'rollouts': {
        const setup = { desc: req.robot, stanceQ: req.stanceQ, ...(req.gains ? { gains: req.gains } : {}) };
        result = req.jobs.map((j) => runRolloutJob(mj, setup, j));
        break;
      }
      case 'pose-rollouts': {
        const setup = { desc: req.robot, stanceQ: req.stanceQ, ...(req.gains ? { gains: req.gains } : {}) };
        result = req.jobs.map((j) => posePolicyRollout(mj, setup, j));
        break;
      }
    }
    post({ id: req.id, type: 'result', result, wallTimeS: (performance.now() - t0) / 1000 });
  } catch (err) {
    post({ id: req.id, type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
}
