import { useEffect, useRef } from 'react';
import type { RobotDescription } from '../../core/model/types';
import { RobotView } from '../../render/robotView';
import { Viewport } from '../../render/viewport';
import type { PhysicsSession } from './session';

export function PhysicsViewport({ session, label = 'Physics mode: MuJoCo simulation with gravity and contact. Rendered from simulator state (ground truth).' }: { session: PhysicsSession; label?: string }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const viewport = new Viewport(host.current!);
    let view: RobotView | null = null;
    let viewDesc: RobotDescription | null = null;
    viewport.start((dt) => {
      session.advance(dt);
      const desc = session.state.description;
      if (desc && desc !== viewDesc) {
        view?.dispose();
        view = new RobotView(desc);
        viewDesc = desc;
        viewport.scene.add(view.root);
      }
      const tr = session.renderTransforms();
      if (view && tr) view.applyTransforms(tr.base, tr.links);
    });
    return () => {
      view?.dispose();
      viewport.dispose();
    };
  }, [session]);
  return (
    <div className="viewport" ref={host} data-testid="viewport">
      <div className="viewport-label">{label}</div>
    </div>
  );
}
