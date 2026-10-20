/** Task lifecycle. A call refused before anything moved is reported as `rejected`. */
export type TaskStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
export type ResultStatus = TaskStatus | 'rejected';

/** A quantity measured in the simulation, in SI units. */
export interface Measurement {
  value: number;
  unit: string;
}

/** What every tool returns to the model and the UI. */
export interface ToolResult {
  taskId: string | null;
  status: ResultStatus;
  reason: string;
  measurements: Record<string, Measurement>;
  fallback: string | null;
}

export function rejected(reason: string, fallback: string | null = null, taskId: string | null = null): ToolResult {
  return { taskId, status: 'rejected', reason, measurements: {}, fallback };
}
