# OpenWiki brief for MoSynth

## Why this wiki exists

MoSynth is a Unity motion-synthesis system that combines motion matching with
neural motion fields, with a PythonNET bridge to a CPython side.

Roughly half of the first-party C# files carry `<remarks>` blocks that are not
API documentation. They are design arguments: they say what was chosen, name the
alternative that was rejected and why, critique the internals of a Unity package,
or cite a measurement taken from this project's own data. That reasoning is
recorded **nowhere else** — not in `AGENTS.md`, not in commit messages.

Those blocks are going to be deleted from the source once this wiki exists, and
replaced with a one-line pointer to the page that carries the argument. **This
wiki is therefore the sole future home of that reasoning.** Treat any design
rationale you find in a comment as a primary artifact to be preserved, not as
background colour to be summarised away.

## The one requirement that matters most

For every non-obvious design decision, the owning page must record:

1. **What was chosen.**
2. **What was rejected, and why it fails.** A decision documented without its
   rejected alternative is not documented — the next reader will re-propose the
   rejected option.
3. **Any concrete number, measurement, or project-specific fact that motivated
   it** (bone counts, measured angles, frame rates, version constraints).
   Preserve these exactly. Do not round, generalise, or drop them.

Put this reasoning on the page that owns the subsystem, in a clearly marked
section. Do **not** create a separate top-level `decisions/` or `rationale/`
directory — rationale belongs with the system it constrains.

## Where the rationale currently lives

Every `<remarks>` block in first-party C# under `Assets/AnimationTools/`,
`Assets/MotionMatching/`, `Assets/MotionField/`, and `Assets/Scripts/` is a
candidate. They appear on private members as well as public ones, so do not
filter by visibility. The densest examples, as orientation rather than an
exhaustive list:

- `Assets/MotionMatching/Runtime/CharacterController/SplinePoseKeypoint.cs`
- `Assets/MotionMatching/Runtime/CharacterController/MotionMatchingControlInput.cs`
- `Assets/MotionMatching/Runtime/Inertialization/Inertialization.cs`
- `Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs`
- `Assets/AnimationTools/Runtime/Evaluation/SplineProjector.cs`
- `Assets/AnimationTools/Runtime/Animation/AnimationClipBaker.cs`
- `Assets/AnimationTools/Runtime/Core/MotionSynthesisComponent.cs`
- `Assets/AnimationTools/Runtime/State/StateSequence.cs`
- `Assets/MotionField/MotionFieldVisualizer.cs`
- `Assets/AnimationTools/Runtime/Python/PythonRuntime.cs`

`Assets/AnimationTools/Tests/Editor/` and `Assets/MotionMatching/Tests/Editor/`
encode expected behaviour and are good evidence for invariants.

## Topics that must not be flattened

These are known to carry subtlety that a generic architecture summary would
lose. Verify each against the source and document it with its reasoning intact:

- **Facing vs. direction of travel.** These are distinct quantities in this
  codebase, and conflating them is a real bug class. Cover how facing is stored,
  why it is not derived from a path tangent, and what the consequence is for
  strafing and backpedalling motion.
- **Spline projection windowing.** Why a windowed nearest-point search is used
  instead of the Unity Splines package's global search, and what goes wrong on a
  self-crossing path.
- **The `MoSynthStage` pipeline contract.** `Init` / `Apply(PoseBuffer, deltaTime)`
  / `OnDestroy`, what a stage may mutate, and the ordering guarantees.
- **Pose representation.** The packed array layout, the two leading frame slots
  and what each holds, the velocity convention, and how the layout survives the
  C#/Python boundary.
- **The PythonNET boundary.** How the runtime is initialised, where the DLL and
  venv paths come from, the Python 3.13 constraint, and what crosses the
  boundary in each direction.
- **The skeleton model.** `Skeleton`, `SkeletonBone`, and `SkeletonBoneOverrides`;
  the bone-ordering and index conventions; the name collision with Unity's own
  `SkeletonBone` and how the codebase handles it.
- **Motion matching query channels**, including how inactive channels are
  handled during search and why that handling was chosen.
- **Clip baking acceptance rules** — which clips are accepted into a database
  and why the threshold is where it is.
- **Inertialization** — what it smooths, and its parameters and failure modes.

## Non-goals

- **Do not restate what the code already says.** A page earns its place only by
  holding something a competent reader could not recover by reading the source:
  reasoning, constraints, cross-file relationships, failure modes.
- Do not narrate the directory tree.
- Do not document vendored code. `External/`, `Assets/Packages/`,
  `Assets/Samples/`, and `Assets/Lib/` are excluded in `.openwikiignore` and are
  not part of this system.
- Do not document `Assets/Local/` or `Assets/LFS/`. `Assets/Local/` is local
  scratch; `Assets/LFS/` is a separate git-lfs repository of source assets kept
  out of this repo because some of its contents are under third-party copyright.

## Relationship to existing docs

`AGENTS.md` at the repo root is the existing canonical orientation document for
coding agents: project overview, code style, tech stack, a directory map, common
tasks, and conventions. **Extend it, do not duplicate it.** Where this wiki goes
deeper on something `AGENTS.md` states briefly, that is correct; where it would
merely restate `AGENTS.md`, link instead.

`.junie/plans/` holds design plans in a Requirements → Technical Design →
Testing → Delivery format and is useful evidence for why a subsystem is shaped
the way it is.

## Audience and tone

Written for an engineer new to the repository who is competent in Unity and C#
but knows nothing about this project's specific choices. Prefer plain prose over
bullet fragments when explaining a mechanism. Be direct about known rough edges,
limitations, and TODOs rather than presenting the system as finished — several
areas are actively in flux, and synthesis methods are not finalised.

## How the wiki is organised

The wiki serves two audiences with deliberately different styles.

**Human-centered pages** — every top-level domain except `agents/`:

- Written for a developer getting to grips with the project. Concept-first,
  plain language, clear and concise, easy to grasp.
- Explain what each system is for and how it behaves at runtime before naming
  files and symbols. Prefer a short Mermaid diagram over a long paragraph when
  it genuinely clarifies a flow or lifecycle.
- No exhaustive inventories, no wall-of-detail invariant lists — link to the
  matching `agents/` page for that depth.

**`agents/` — AI coding agent reference:**

- A deliberate, user-mandated section (not a generic catch-all): dense,
  operational, evidence-heavy material that helps AI coding agents work on this
  repository safely and effectively.
- Organised as **one subfolder per subsystem** (e.g. `agents/animation-tools/`,
  `agents/motion-matching/`, `agents/motion-field/`, `agents/python/`,
  `agents/tooling/`), mirroring the human domains where sensible.
- Content style: exact invariants and rules with the precise APIs involved,
  known hazards and traps (including "do NOT fix this" notes), editing and
  verification workflows, per-subsystem internals, exact file/symbol references.
- It is fine — expected — for `agents/` pages to overlap topically with human
  pages; they differ in depth and audience, and each human page should link to
  its `agents/` counterpart where one exists.
- The section does not exist yet. It is built incrementally: create the
  subfolder the first time there is something operational worth keeping, rather
  than restructuring the existing pages up front.

## Ownership and when to edit

- `agents/` is **owned by the coding agents**. They write to it freely, without
  asking, whenever they learn something operational worth keeping.
- Every other section is human-facing and should read like documentation, not
  like agent notes.
- The wiki is updated as part of finishing a task, and the wiki change is
  committed alongside the code change. A behaviour change that leaves its page
  stale is not finished.
- Any agent asked to organise, refresh, or restructure the wiki follows these
  same guidelines and the comment-placement rules in root `AGENTS.md`.

Root `AGENTS.md` stays the canonical short agent brief; the `agents/` wiki
section carries the deep detail behind it.
