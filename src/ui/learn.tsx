import { createContext, useContext, type ReactNode } from 'react';

/** True while learn mode is on: panels then show explanations of what they compute. */
export const LearnContext = createContext(false);

/** An explanation block that appears only in learn mode. */
export function Explain({ children }: { children: ReactNode }) {
  const on = useContext(LearnContext);
  if (!on) return null;
  return (
    <div className="explain" role="note" aria-label="Explanation" data-testid="explain">
      {children}
    </div>
  );
}
