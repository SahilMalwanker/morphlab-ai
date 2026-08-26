import { parseCommand, type CommandContext, type ParseResult } from './parser';

export interface Interpretation {
  result: ParseResult;
  source: 'local rules';
  note: string | null;
}

/**
 * Text-to-command interpreter with the local rules (Tier A, works offline).
 * Language-model interpretation goes through the AI proxy and the validated
 * tools in `src/ai`; no provider endpoint or key is ever handled in the page.
 */
export class CommandInterpreter {
  interpret(text: string, ctx: CommandContext): Promise<Interpretation> {
    return Promise.resolve({ result: parseCommand(text, ctx), source: 'local rules', note: null });
  }
}
