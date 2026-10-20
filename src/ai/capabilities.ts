export type BodyKind = 'arm' | 'legged' | 'wheeled';

/** What the current robot and scene support; the tools offered to a model are derived from it. */
export interface RobotCapabilities {
  name: string;
  morphologyHash: string;
  bodyKind: BodyKind;
  jointCount: number;
  targetableLimbs?: readonly string[];
  /** Saved skills evaluated on this morphology. */
  skills: readonly string[];
  /** Places remembered on the current map (world memory). */
  places: readonly string[];
  /** Command ranges of the body's learned walking controller (D28), when one is trained. */
  walking?: { forwardMps: readonly [number, number]; leftMps: readonly [number, number]; turnRadps: readonly [number, number] } | null;
  features: {
    pointing: boolean;
    navigation: boolean;
    sceneObjects: boolean;
    demonstrations: boolean;
    controllerExperiments: boolean;
    skillEvaluation: boolean;
    /** The AI may compose a skill draft (poses and goals), try it, and evaluate and save it. */
    skillAuthoring: boolean;
    /** Saved skills can run one after another. */
    skillSequences: boolean;
    /** The view's robot can be switched to a preset or the Build tab's robot. */
    robotChoice: boolean;
    /** Pushes and resets of the Physics tab simulation. */
    physicsTools: boolean;
    /** Inverse kinematics and joint angles in the Kinematics tab (geometry, no physics). */
    kinematicsTools: boolean;
    /** Robot design in the Build tab. */
    designTools: boolean;
  };
}
