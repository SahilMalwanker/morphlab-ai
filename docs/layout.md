# MorphLab user interface: layout

Source reference reviewed on 2026-10-16, from
[src/ui/App.tsx](../src/ui/App.tsx),
[src/ui/WorkspaceNav.tsx](../src/ui/WorkspaceNav.tsx),
[src/ui/theme.css](../src/ui/theme.css) and the mode components. It
describes the Skills-first workspace (D34) and the D35-D38 additions;
verification evidence and findings are recorded in [status.md](status.md).
Technology is in [ui.md](ui.md) and behavior in [interface.md](interface.md).

Quick navigation: [grid](#1-application-grid), [breakpoints](#2-responsive-behavior),
[panel anatomy](#3-panel-anatomy), [viewport](#4-viewport-conventions),
[view inventory](#5-views), [comparison](#6-comparison-and-plots),
[command area](#7-command-area-and-vertical-space),
[state placement](#8-where-feedback-belongs), [guardrails](#9-layout-guardrails),
[review matrix](#10-layout-review-matrix).

## 1. Application grid

The shell (`.app`) has two columns: the navigation and the workspace. The
workspace stacks the notice area, the lab header (lab tools only), the
work area and the command bar. The open view places its parts into the
work area's named grid areas. The diagram is schematic.

```text
+------------+-------------------------------------------------------------+
| MorphLab   | notice: getting-started guide (optional), notices           |
|            +-------------------------------------------------------------+
| AI         | lab head (lab tools only): Lab tools / Physics  [Back]       |
| Playground +-------------------------------------------------------------+
|  Skills... | work area (the open view, see section 5)                     |
|            |                                                             |
| LAB TOOLS  |                                                             |
| Build      |                                                             |
| Kinematics |                                                             |
| Physics    |                                                             |
| Control lab|                                                             |
| Navigate   +-------------------------------------------------------------+
|            | command: Command [.......] Run Stop-task status [Voice/AI]  |
| Guide      |   options row (when open), bounded output                   |
| Learn mode |                                                             |
+------------+-------------------------------------------------------------+
```

| Element | Size | Content |
|---|---|---|
| `nav.sidenav` | `--nav-w` (188 px, 164 px at 1200 px and below) | Brand, AI Playground, Lab tools group, Guide, Learn mode, units note |
| `.notice-area` | auto | `Guide` and `Notices`; empty when hidden and nothing needs attention |
| `.lab-head` | auto, at least 40 px | Lab tool title and Back to AI Playground; not rendered in the Playground |
| `main.workarea` | `minmax(0, 1fr)`: the rest | The open view; labeled with its title; `tabindex="-1"` so a view switch can move focus to it |
| `footer.commandbar` | auto, output bounded | Command row, Voice and AI options, output |
| `aside.pg-side` (Playground only) | `--pg-side-w`, beside the work area and command bar | Teach, Library or Attempts; rendered into `.pg-side-slot` after the command bar |

Work area and Playground templates:

```css
/* lab tools */
.workarea { grid-template-columns: 300px minmax(320px, 1fr) 340px; grid-template-areas: 'left view right'; }
/* AI Playground: the work area holds header, status and view; the side panel is a workspace item */
.workarea.mode-skills { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto auto minmax(0, 1fr); grid-template-areas: 'head' 'status' 'view'; }
.workspace.pg {
  grid-template-columns: minmax(0, 1fr) var(--pg-side-w);  /* clamp(320px, 28vw, 400px) */
  grid-template-areas: 'notice notice' 'labhead labhead' 'work side' 'command side';
}
.workspace.pg.side-collapsed { grid-template-columns: minmax(0, 1fr); grid-template-areas: 'notice' 'labhead' 'work' 'command'; }
```

- The work area is a real grid container (`<main>`); the earlier
  `display: contents` tab panel is gone.
- The Playground side panel is rendered through a React portal into
  `.pg-side-slot` (`display: contents`), which follows the command bar in
  the document. On wide windows it spans the height of the work area and
  the command bar; in one column it comes after the command bar, so the
  document order, the Tab order and the screen order agree.
- Panels scroll vertically on their own; the viewport hides overflow and
  fills its cell. Above 900 px the page itself does not scroll.
- Side widths are fixed or clamped CSS lengths, not draggable split panes
  (drag resizing is blocked, see [status.md](status.md)). The Playground
  side panel collapses and restores with a button.

## 2. Responsive behavior

| Window width | Navigation | Work area | Scrolling |
|---|---|---|---|
| Above 1300 px | Left column, 188 px, units note | Lab tools 300 / `1fr` / 340 px; Playground view plus side panel up to 400 px | Panels; Control lab results; command output |
| Above 1200 px through 1300 px | Same, units note hidden; guide in two columns | Same | Same |
| Above 1000 px through 1200 px | Left column, 164 px | Lab tools 250 / `minmax(300px, 1fr)` / 290 px | Same |
| Above 900 px through 1000 px | Top bar: brand, current view, Menu button (disclosure) | Same tracks; Playground side panel `clamp(280px, 36vw, 360px)` | Same |
| 900 px and below | Top bar with Menu | One column, view `55vh`; lab tools: view, right, left, then the command bar; Playground: header, status, view, command bar, side panel | Page flow |

```text
+----------------------------------+
| MorphLab  AI Playground   [Menu] |
|  (Menu open: views, Guide, Learn)|
+----------------------------------+
| Guide and notices, when present  |
+----------------------------------+
| Playground header (wraps)        |
| Job and message strip            |
| View: 55vh                       |
+----------------------------------+
| Command row, options, output     |
+----------------------------------+
| Side panel (unless hidden)       |
+----------------------------------+
```

The Menu is a disclosure in the page flow, not a modal drawer: it does not
trap focus, closes with Escape or a choice, and returns focus to the Menu
button. Lab tools keep the CSS reorder (right panel before left panel);
the DOM order is unchanged, so keyboard order must be reviewed separately.
The command bar is not fixed or sticky; in the Playground it comes right
after the robot view, so Stop task is never below a long teaching form.
On desktop widths above 900 px with a window height of 650 px or less,
the app has an 800 px minimum document height. The page scrolls rather
than allowing the header and command bar to collapse the robot view.

## 3. Panel anatomy

```text
aside.panel
  section.section          one per topic, separated by a 1 px line
    h2                     17 px, sentence case, primary text
    .field                 label above the input
    .row                   fields or buttons side by side
    .readout               two-column definition list, values in monospace, right-aligned
    .joint                 slider with name, value (amber at a limit) and range
    .note / .note.warn / .note.error
    .explain               learn-mode explanation, accent left border
```

Primary actions use `.btn.primary` (mint fill, dark text); the Stop button
uses `.btn.stop` (muted red fill, red outline and text).

| Element | Current sizing and treatment |
|---|---|
| Navigation item | At least 36 px high, text label; the active Playground has a mint fill, active lab tools an inset accent marker |
| Lab header | At least 40 px; title and Back to AI Playground |
| Side panel | 12 px top / 16 px sides and bottom; independent vertical scrolling on desktop |
| Section | 12 px vertical padding, bottom separator except on the last section; no decorative outer card |
| Field | Label/input stack with a 2 px gap and 8 px bottom margin |
| `.row` | Flex items separated by 8 px; no global wrapping rule |
| Readout | Shrinkable `1fr 1.4fr` columns, 4 px row gap, 12 px column gap; wrapping labels and right-aligned tabular monospace values |
| Button | `--control-h` (32 px) minimum height, 13 px text, 12 px horizontal padding, 6 px radius; `.primary` solid mint, `.danger` red text for Delete and Discard, `.stop` red outline for Stop task |
| Status | 13 px label and an 8 px colored dot; text carries meaning as well as color |
| Focus | 2 px accent outline with 2 px offset |
| Learn-mode explanation | Accent left border and inset background; expands its containing section |

Use the [theme token reference](ui.md#3-styling) for colors and fonts.
The current interface uses native inputs, checkboxes, selects and mostly
text buttons. It has no installed icon/component library to assume when
editing an existing control.

## 4. Viewport conventions

- A label in the top-left corner (`.viewport-label`, 12 px, translucent
  background, ignores the pointer) states what the view shows and where
  its data comes from. Exact texts:

| View | Label |
|---|---|
| Build | Builder preview: home configuration computed by forward kinematics. No physics. |
| Kinematics | Kinematics mode: poses computed by forward kinematics. No physics. |
| Physics | Physics mode: MuJoCo simulation with gravity and contact. Rendered from simulator state (ground truth). |
| Control lab | Recorded MuJoCo runs, plotted after each run completes. Positions come from joint encoders. |
| Skills | Skills: live MuJoCo simulation. Controllers see joint encoders, IMU and foot touch only; rendering uses simulator state. |
| Skills, comparison | Replay of recorded simulator states, display only: nothing is simulated. Left: A, right: B. |
| Navigate | Navigate: MuJoCo simulation. Rays, map and path come from the controller; the robot is drawn from simulator state; purple markers are remembered places. Inset: virtual camera (display only). |

- 3D scenes: dark ground plane, 10 m grid in XY, Z up, orbit camera.
- Navigate draws an inset from a virtual camera just ahead of the robot's
  front in
  the bottom-right corner (12 px margin, grey 1 px frame), 300 x 180 px or
  at most 40 % of the viewport width; it is not drawn when narrower than
  120 px. Overlays (rays, grid, path, goal, places) are on a separate
  layer that the inset camera does not see.

Control lab overrides its label to be inline above the plots, rather than
absolutely positioned over a canvas. In the other modes the label begins
12 px from the top/left; it must remain legible without intercepting
camera or target interactions.

The shared [Viewport](../src/render/viewport.ts) sizes its canvas from the
host element through `ResizeObserver`, updates camera aspect, and caps
device pixel ratio at 2. A nonzero host area is required: a visible canvas
element alone does not prove that a correctly framed scene rendered.
The renderer and canvas plots also contain their own color literals;
changing CSS tokens alone does not retheme them.

## 5. Views

Section titles are given as they appear on screen; "(n)" is a count.
Columns describe independent top-to-bottom ordering, not exact horizontal
alignment between sections. Conditional sections appear only with the
appropriate data; this inventory does not require placeholders for them.
The AI Playground is described last because it uses its own template.

### Build

Source: [src/ui/builder/BuilderMode.tsx](../src/ui/builder/BuilderMode.tsx).

| Left: "Robot builder" | View | Right: "Limb inspector" |
|---|---|---|
| Start from a preset | Robot at its home configuration (forward kinematics) | Validation errors or notice (top, when present) |
| Legged generator | | Limbs |
| Arm generator | | Joints (for the selected limb and joint) |
| Body | | Morphology version |
| | | Export and import |
| | | Use this robot |
| | | Saved robot versions (n) |

Generators and Body edit the overall design on the left. The right panel
edits the selected limb/joint, then shows provenance and hand-off actions.
Validation errors precede the inspector sections. Mirror controls depend
on a partner limb; saved versions depend on available storage.

### Kinematics

Sources: [src/ui/kinematics/KinematicsMode.tsx](../src/ui/kinematics/KinematicsMode.tsx),
[src/ui/kinematics/ChainEditor.tsx](../src/ui/kinematics/ChainEditor.tsx).

| Left: "Robot" | View | Right: "Inspector" |
|---|---|---|
| Robot | Robot, IK target sphere with a translate gizmo, dashed residual line; optional frames, joint axes, manipulability ellipsoid, workspace samples | Inverse kinematics |
| Serial chain (editable chains only) | | Joints |
| Overlays | | Tip position (world frame) |
| | | Jacobian |

Robot/chain selection belongs on the left; target coordinates, solver
status and the resulting joint/tip readouts belong together on the right.
The target gizmo and optional overlays stay in the same unframed scene.
Keep the numeric target fields reachable without dragging the gizmo.

### Physics

Source: [src/ui/physics/PhysicsMode.tsx](../src/ui/physics/PhysicsMode.tsx).

| Left: "Simulation setup" | View | Right: "Simulation inspector" |
|---|---|---|
| Robot | Live MuJoCo simulation | Simulation (status badge, Run or Pause, Reset, readouts, MuJoCo warnings) |
| Joint PD controller (kp 0 to 150, kd 0 to 8, stability readouts) | | Joint (selector, commanded position slider, position and effort time plots) |
| Push tool (impulse 0.5 to 8 N*s, +X, -X, +Y, -Y) | | Events |

The simulation status and Run/Pause/Reset controls lead the right panel.
The selected joint's target and plots follow them; the event log is last.
An instability warning must stay associated with the simulation rather
than looking like an ordinary successful command response.

### Control lab

Source: [src/ui/control/ControlMode.tsx](../src/ui/control/ControlMode.tsx).

| Left: "Experiment setup" | View: "Step response plots" | Right: "Challenges" |
|---|---|---|
| Experiment | Scrollable column: progress, joint position plot (260 px), joint effort plot (220 px), stale-result warning, metrics table (A and B, measured and linear), saturation and model notes | Challenges (title, goal, Run, criteria per gain set) |
| Gain set A | | |
| Gain set B | | |

Control lab uses the middle area as a scrollable results document, not a
3D scene. Skill comparison has its own scrolling control strip below its
canvas. Do not replace either with a decorative viewport placeholder.

### AI Playground (Skills mode)

| Area | Content |
|---|---|
| `head` | Title "AI Playground" with "Skills"; Robot select; body facts (free-standing or fixed base, limbs, joints) and the morphology hash; engine badge (Loading physics engine, Live simulation, Simulation unavailable); side panel buttons Teach, Library (n), Attempts (n); Hide panel or Show panel |
| `status` | Job label, percent, progress bar, detail and its Cancel; the latest message. Rendered only when there is something to show, never hidden with the side panel |
| `view` | Live MuJoCo robot, or the attempt comparison; an engine notice overlays it while loading or after a failure |
| `side` (workspace grid) | One of Teach, Skill library, Attempts; hidden when collapsed. Beside the work area and command bar on wide windows; after the command bar in one column |

Side panel content:

| View | Ordered content |
|---|---|
| Teach, workflow buttons | Teach a skill, Foot lift example, Walk by command |
| Teach, no draft | Teach a skill: explanation, New skill, Explore the foot lift example, Open the skill library (n) |
| Teach, keyframes | Teach a skill (name, draft state, Start over, Mirror, Discard); 1. Motion (Pose the robot, Keyframes (n), playback note); 2. Success criteria (n); 3. Try (Try once, last playback); 4. Evaluate and save; Improve or train (optional), collapsed |
| Teach, gait | Same, with the gait parameters in 1. Motion |
| Foot lift example | Example skill: lift one foot; 1. State machine (hand-designed); 2. Tuned parameters; 3. Demonstration and replay; 4. Imitation learning; 5. Reward-based improvement |
| Walk by command | Walk by command; Command when a walking controller exists |
| Library | Skill library (n): Export library, Import skills; one record per skill (name, method, trials and compatibility, Playing and Unverified here badges); the open record shows the full evaluation readout, Play, Re-evaluate here and Delete |
| Attempts | Attempts (n): last playback, attempt list, A and B selectors, Compare side by side |

The newest library record opens by default and after each new save or
import; the user's choice otherwise stays. At most one record is open.

Sources: [src/ui/skills/SkillsMode.tsx](../src/ui/skills/SkillsMode.tsx),
[src/ui/skills/TeachPanel.tsx](../src/ui/skills/TeachPanel.tsx),
[src/ui/skills/WalkPanel.tsx](../src/ui/skills/WalkPanel.tsx),
[src/ui/skills/LastPlay.tsx](../src/ui/skills/LastPlay.tsx).

### Navigate

Source: [src/ui/nav/NavMode.tsx](../src/ui/nav/NavMode.tsx).

| Left: "Navigation setup" | View | Right: "Navigation status" |
|---|---|---|
| Robot and map (Robot with disabled entries and their reasons, drive geometry, Test map, Controller inputs) | Arena, obstacles, occupancy grid on the floor (green free, red occupied), planned path, range rays, goal, remembered places, robot; inset camera bottom right | Status (readouts: simulated time, driving, goal, ground-truth and estimated position, speed, replans) |
| Drive (Drive to goal, Stop, Reset, goal fields, speed limit, Remote control, key list) | | Range rays (sensor) |
| Evaluate all maps (run, cancel, result table) | | Remembered places (n) |

The arena is the primary view; the virtual-camera inset is subordinate.
Numeric goals and speed controls stay in Drive. Remembered-place markers
use the same scene as the robot, while names, coordinates, provenance,
Drive here and Forget remain in the right panel.

## 6. Comparison and plots

Attempt comparison replaces the Skills middle area; it does not add a
second application frame or two independently running simulations.

```text
+------------------------------------------+
| Replay label                             |
|                                          |
| Recorded robot A     Recorded robot B     |
|                                          |
+------------------------------------------+
| Play/Pause/Replay  clock  Close comparison|
| Replay time slider                       |
| A result                                 |
| B result                                 |
| Comparison plot                          |
+------------------------------------------+
```

The `.compare-canvas` flexes with a 200 px minimum height.
`.compare-controls` scrolls and has `max-height: 55%`; its plot is 150 px
high. Both recordings share a timeline. These constraints can compete on
a very short viewport and need a visual check, not an assumed guarantee
that both regions fit.

| Plot | Placement | Data/update behavior |
|---|---|---|
| `TimePlot` | Physics joint inspector | Canvas 2D, rolling telemetry window; redraw timer independent of React snapshots |
| `StaticPlot` in Control lab | Middle results area, position 260 px and effort 220 px | Recorded series after an experiment, with measured and model traces distinctly labeled |
| `StaticPlot` in comparison | Below replay controls | Recorded attempt metric, not live physics |

Plot titles carry units; legends identify traces, with dashed lines where
specified. The current plots are displays, not a charting-library zoom,
selection or export toolbar. Keep nearby readouts and explanations rather
than assuming every plotted value is available to assistive technology.
Sources: [src/ui/plot.tsx](../src/ui/plot.tsx) and
[src/ui/skills/CompareView.tsx](../src/ui/skills/CompareView.tsx).

## 7. Command area and vertical space

[src/ui/CommandBar.tsx](../src/ui/CommandBar.tsx) fills the `command` area:

1. Command row: label and field, Run or Send, Stop task, Running (while
   busy), the AI status badge and the Voice and AI button.
2. Voice and AI options (a disclosure, hidden by default, kept open while
   the microphone listens): Speak or the reason voice input is missing,
   Check AI with the reason when AI is unavailable, Read replies, Forget
   conversation.
3. Output: consent and voice messages, Stop and live-change messages, the
   result with its source, and the collapsible What the AI did trace.

The output scrolls inside `max-height: clamp(3.5rem, 24vh, 14rem)`, so an
expanded trace or long reply cannot squeeze the robot view; the latest
result line comes first. On narrow screens the page scrolls and the
command bar is not sticky; in the Playground it comes right after the
robot view and before the side panel, in lab tools after their panels.

## 8. Where feedback belongs

| State or message | Existing location and purpose |
|---|---|
| Browser or storage availability | Notice area below the header; global to the workspace |
| Builder validation | Top of the right inspector, before the editable sections |
| Numeric field error | The affected field's invalid styling and the owning mode's explanation |
| IK outcome | Inverse kinematics section next to target coordinates and residual |
| Simulation status/instability | Physics Simulation section and warnings |
| Experiment progress/stale setup | Control lab middle results area |
| Training/evaluation progress | Playground status strip under the header, or Navigate batch section, with its Cancel action |
| Skill evaluation/provenance | The individual library record, not a global success banner |
| Command/AI outcome | Bottom response region; model words separated from tool measurements |
| React mode error | ErrorBoundary fallback in the view area, leaving shell navigation available |

An empty library, no draft or no comparison attempts is an ordinary
empty state. Show its existing explanation/action without inventing
example success records, fake charts or placeholder capabilities.

## 9. Layout guardrails

These are review requirements, not additional CSS implemented by this doc.

- Preserve the workbench hierarchy: setup at left, main scene/results in
  the middle, inspection/library at right. Keep whole page sections
  unframed; do not turn every section into a nested card.
- Use theme spacing and type sizes. Do not enlarge a compact panel heading
  into hero text or use tiny type to hide an overflow problem.
- Let long robot names, failure reasons and trace text wrap where needed.
  Check repeated action groups, selects and readouts with real long values.
  `.row` does not wrap automatically, and no universal long-word rule exists.
- Preserve `min-width: 0`/`min-height: 0` on shrinking grid/flex children.
  Give new canvases a stable host area and dispose them when removed.
- Keep labels clear of gizmos, camera controls and the Navigate inset.
  Data-source labels are part of the interpretation, not decoration.
- Preserve visible focus, native input semantics and numeric alternatives
  to pointer actions. A visual reorder must not be assumed to reorder Tab.
- Review CSS cascade as well as media-query intent. Responsive rules now
  sit at the end of `theme.css`, so the narrow-window panel borders apply
  as written.

There are no draggable split panes, dockable panels, modal drawers or
touch driving controls in this layout. A responsive stack is not proof of
a touch-first or fully accessible mobile workflow.

## 10. Layout review matrix

Suggested checks for UI changes; the 2026-10-16 audit results for the
current build are recorded in [status.md](status.md):

| Viewport or condition | Inspect |
|---|---|
| 1440 x 900 | Navigation column, Playground view width with the side panel open and hidden, default guide and command output |
| 1100 x 800 | Compact navigation and lab sidebars, long control labels and two-column guide |
| 1301/1300, 1201/1200, 1001/1000, 901/900 px widths | Exact transitions for metadata, navigation, guide and work-area tracks |
| 740 x 900 | Existing narrow-window test size; Menu reaches every view and no horizontal page overflow |
| 390 x 844 | Small-screen stack, all command buttons, long skill names and large notices |
| Short desktop window or 200% browser zoom | Header/footer growth, useful scene height and accessible scroll regions |
| Guide + Learn mode + expanded AI trace | Maximum ordinary text expansion without hidden controls |
| Skills comparison and Navigate inset | Both recordings framed, shared timeline visible, camera inset distinct from overlays |

- [ ] Record the CI build/artifact and browser; do not inspect a stale build after source changes.
- [ ] Capture relevant desktop and narrow screenshots; check canvas content, not only its bounding box.
- [ ] Check long names, warnings, loading, busy, disabled and empty states, not only the ideal populated screen.
- [ ] Traverse controls with the keyboard and verify focus remains visible in the correct scroll container.
- [ ] Inspect console/page errors and resource cleanup after repeated mode switches.
- [ ] Record measured results and remaining gaps in [status.md](status.md).

Existing coverage lives in [tests/e2e/a11y.spec.ts](../tests/e2e/a11y.spec.ts),
[tests/e2e/polish.spec.ts](../tests/e2e/polish.spec.ts) and mode-specific
specs. The 740 px test does not certify every smaller width. Follow the
[verification contract](ui.md#11-verification-contract).
