---
type: Agent Reference
title: Root motion and control input hazards
description: Operational rules for editing the trajectory origin, the root-follow stage and the ordering between control inputs and the synthesis tick.
tags: [agents, motion-matching, root-motion, control-input, hazards]
generated: {by: "claude-code", at: "2026-09-06T12:33:15.598Z"}
---

# Root motion and control input hazards

Operational notes for changing how a character is steered or placed. Concepts and rationale are on
[root following](../../animation-tools/root-following.md) and
[control inputs](../../motion-matching/control-inputs.md); this page is the list of things that
break quietly.

## Stage order is a correctness constraint, not a preference

```
MotionMatchingStage → Inertialization → RootFollowStage   (→ FootLockStage, when it exists)
```

- **`RootFollowStage` before `Inertialization`** and the blend smooths the correction away. The
  character then still lags, and nothing reports a problem.
- **A foot-locking stage must be last**, because it asks `ComputeAppliedFrame` where the pose will be
  rendered, and any later stage that touches bone 0 invalidates that answer.

`StageEnabledOverride` matches on the **type name without namespace**, as a string
(`"RootFollowStage"`). Renaming a stage type silently turns every override that names it into a
warning at spawn time, and the benchmark method then measures the un-overridden prefab.

## Three ways `RootFollowStage` goes inert

It warns and does nothing when the target is null, when the target is a child of the character
(following yourself), or when **`rootPositionsMask` is off** — the component never integrates
anything then, so a corrected velocity has nowhere to go. Check the console before concluding the
maths is wrong.

## Do not write the character's Transform from a stage

A correction goes into bone 0's velocity channels, through
`SimulationFrame.DecomposeRootVelocity` / `RecomposeRootVelocity`. Writing `owner.transform` directly
appears to work and then fights the integration in `ApplyPoseToSkeletonTransforms`, which runs after
every stage and advances the Transform from the pose regardless.

The one sanctioned exception, should height following ever be added, is `position.y` — the frame is
ground-projected, so y genuinely cannot travel through bone 0.

## `SimulationFrame` is shadowed inside MotionSynthesisComponent

`MotionSynthesisComponent` has a property named `SimulationFrame` of type `SimulationFrameDef`, which
shadows the static class. **Every call site inside that file must read
`AnimationTools.SimulationFrame.Compute(...)`.** This bites on any new call site added there, and the
error message does not point at the shadowing.

## Anchored and following do not compose

`AnchoredDirectionControlInput` has no simulation object, so there is nothing for `RootFollowStage`
to follow. Pointing the stage at the input's own Transform is not "both mechanisms" — it is a stage
following a Transform nothing drives.

The valid pairs are: free simulation object alone; anchored input alone; free simulation object plus
`RootFollowStage`.

## `alignTargetOnInit` vs. benchmark placement

`RootFollowStage` moves a plain Transform target onto the character at `Init`, because the shipped
prefab authors them about two metres apart and a zero half-life would teleport the character there.
It is **skipped when the target implements `IFrameTarget`**, since such a target owns its position.

`SynthesisBenchmarkDriver.PlaceAtPathStart` positions the character while the instance is still
inactive, so `Init` sees the placed frame. Anything that moves a character after `Awake` has to
account for the alignment already having happened.

## Editing a control input

- The base class ticks in **`Update`**, deliberately, so the trajectory a search runs against is the
  current frame's. Do not move it back to `LateUpdate`.
- `ResolveTrajectoryFeaturePair` asserts both named channels exist **and predict at the same
  horizons**. A database whose position and direction channels disagree fails there rather than
  producing a misaligned query.
- Convert to the character frame through `WritePlanarPosition` / `WritePlanarDirection` on the base
  class rather than calling `InverseTransformPoint` inline. `SplinePoseKeypointControlInput` is the
  one deliberate exception — it keeps absolute Y and ignores scale, and says so.
- **Negative prediction frames are the past.** A spring cannot answer them (running it with a
  negative timestep does not run it backwards), so a spring-driven input records a
  `TrajectoryHistory`; a path follower reads its own path behind it.

## Do not "fix" the open-loop spline inputs

`SplineControlInput` and `PfnnSplineControlInput` advance on their own clock and never look at the
character. That is what makes drift measurable as path-following error, and it is what the benchmark
depends on. Steering them toward the character would silently invalidate every reported `trajErr`.

Their `IFrameTarget` implementation is not a change to that: it reports where the point on the path
is, and only a `RootFollowStage` deliberately added to a prefab acts on it.

## Verification without the Unity Editor

The Editor is not always running, and neither Rider's build verdict nor the Unity console is
trustworthy for compile errors (`Editor.log` is). Unity ships a usable Roslyn:

```
C:\Program Files\Unity\Hub\Editor\6000.4.4f1\Editor\Data\NetCoreRuntime\dotnet.exe
C:\Program Files\Unity\Hub\Editor\6000.4.4f1\Editor\Data\DotNetSdkRoslyn\csc.dll
```

Compile an assembly by taking its **references** from its generated `.csproj` (resolving each
`ProjectReference` to `Library/ScriptAssemblies/<name>.dll`, which Unity has already built) but its
**source list from the asmdef's folder on disk** — the `.csproj` file list is only as fresh as the
last Editor regeneration, so it omits new files and still lists deleted ones.

Pure fixtures (`RootFollowTests`, `TrajectorySteeringTests`) can then be run by reflecting over the
built test assembly. Anything touching a `GameObject`, an asset, or a `PoseBuffer` still needs the
Editor's test runner via `MoSynth/Tests/Run EditMode Tests`, which writes
`Temp/animtools_test_results.txt`.
