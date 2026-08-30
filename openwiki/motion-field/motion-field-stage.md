---
type: Architecture Guide
title: The motion field stage
description: Driving a character from a neural motion field evaluated in Python — the four policies, why a value function beats greedy, and how the goal is expressed.
tags: [motion-field, stage, policy, pythonnet]
sources:
  - id: openwiki-source-c9ec04c146f7125c2e260739
    resource: repo://Assets/AnimationTools/Runtime/Core/MoSynthStage.cs
  - id: openwiki-source-1fb0416756719f48922c424e
    resource: repo://Assets/MotionField/MotionFieldSplineControlInput.cs
  - id: openwiki-source-13742752b942a8c72fc71381
    resource: repo://Assets/MotionField/MotionFieldStage.cs
  - id: openwiki-source-556de75b4b36254c0d3e2158
    resource: repo://Python/MotionField.py
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
---

# The motion field stage

`MotionFieldStage` drives the character from a neural motion field **evaluated in Python**, embedded
in-process via PythonNET. Each frame it asks the field for the action with the best long-term value,
given where the player wants to go.

It is an ordinary [`MoSynthStage`](../animation-tools/synthesis-pipeline.md), so it is
interchangeable with [motion matching](../motion-matching/matching-stage.md) at the stage level.

## Why a value function

That look-ahead is what a value function buys.

**A one-step-greedy policy only asks which action points closest to the goal on the very next
frame.** Turning a walk cycle takes several steps to set up, so a greedy policy oscillates rather
than committing to a turn — it keeps picking whatever momentarily reduces the heading error and never
commits to the sequence that would actually fix it.

Scoring each candidate with `Q = R + γ·V(s′)` instead means the policy will **accept a worse
immediate heading when it sets up a better turn**. That anticipation is the entire reason for
training a value function.

Without a trained one, the stage falls back to exactly the greedy policy it was designed to improve
on.

## The four policies

| `Policy` | Python method | Purpose |
| --- | --- | --- |
| `Optimal` (default) | `optimal_action` | trained value function when available, greedy otherwise |
| `Greedy` | `greedy_action` | force one-step-greedy, ignoring any trained value function |
| `Playback` | `get_next_pose` | debug: play the database back frame by frame |
| `NearestNeighbour` | `get_next_pose_from_field` | debug: snap to the successor of the nearest state |

**Policy dispatch lives in C#**, as a `switch` that names a *different Python method* per case. So the
policy never crosses the boundary as a value — no enum, no string, no integer is sent. An unknown
policy is a C# `NotImplementedException` rather than a Python `AttributeError`.

## Theta: the goal, expressed relative to the character

The control input supplies a desired world heading. The stage converts it to `Theta` — **the
character's own heading expressed in the goal frame**.

```csharp
Theta = -Vector3.SignedAngle(forward, DesiredWorldDirection, Vector3.up) * Mathf.Deg2Rad;
```

That framing is what makes it work across the boundary. **Python accumulates its root pose
independently of Unity's transform**, and the only thing coupling the two is the per-step yaw.
Measuring the goal relative to the character's *current* facing means the two accumulators never have
to be synchronised at all.

The negation matches the training convention, where reward is `-|theta + delta_yaw|`: the action that
cancels theta is the one that turns the character onto the goal.

Callers are expected to push a fresh direction every frame, because Theta is measured against the
root's current facing. A near-zero vector keeps the previous heading and only refreshes Theta.

The character Transform's +Z is the facing, because
[the pipeline](../animation-tools/synthesis-pipeline.md) keeps it aligned with the yaw-only
simulation frame the pose derives.

## Control inputs

`MotionFieldControlInput` is the abstract base; subclasses supply a desired world heading each frame
and the base pushes it into the stage *before* the synthesis tick, so the goal angle is measured
against the root pose the stage is about to step from.

The stage is resolved **lazily**, because stages are populated in the synthesis component's `Awake`
and this component must work regardless of `Awake` ordering.

Two concrete inputs:

- **`MotionFieldDirectionControlInput`** — stick/WASD, camera-relative. Small input means "keep the
  last heading", which matches the fact that a released stick may never report zero — and the policy
  cannot stand still anyway.
- **`MotionFieldSplineControlInput`** — pure pursuit along a spline. It reports **`float.NaN` as its
  target speed**, because the policy has no speed control at all. Lookahead distance is therefore its
  only tuning knob: shorter hugs the spline but can oscillate, longer cuts corners.

That NaN is honest rather than a placeholder, and it has consequences downstream: it is why
[lap counting measures actual progress](../animation-tools/path-following-metrics.md) rather than
deriving a lap time.

The spline input tracks its position with a [continuity-preserving projector](../animation-tools/path-following-metrics.md),
because on a self-crossing path the globally nearest point flips branches at the crossing and takes
the steering target with it. It resets that projector whenever the spline changes — a benchmark sweep
hands it a different spline every run.

It returns the vector **from the character to its lookahead target**, which has a consequence at the
end of an open spline: the target stops advancing, the vector shrinks toward zero, and the stage reads
that as "no opinion" and keeps the previous heading. Combined with the early-outs for a null container,
a spline with fewer than two knots, or a sub-millimetre length — all of which return zero — a
degenerate path never produces a wild heading, only a held one.

## Startup

`Init` bootstraps the interpreter through [`PythonRuntime`](python-interop.md), imports the two
modules, loads the database, constructs the Python `MotionField` object with the config's
hyperparameters, then gates the value function three ways:

| Condition | Result |
| --- | --- |
| file missing | warn, run greedy, "Press Train Motion Field on the config to fix this" |
| file present but config changed since training | warn, run greedy |
| otherwise | load it |

**None of the three is fatal.** A stale or absent value function degrades to greedy control rather
than throwing, because failing hard here would surface as an opaque managed exception in a player
build.

Refusing a stale one is not caution. **The value function is indexed by database row**, so a
mismatched pair reads plausible numbers off the wrong poses — and the character merely moves badly
rather than visibly failing. See [config and training](config-and-training.md) for the full staleness
contract.

If `reloadPythonModules` is set, the module cache is invalidated **once, up front**, so every module
imported afterwards comes from the same generation of the source and shares one copy of each class.

## Debug data

When `collectDebugData` is on, the stage pulls the chosen neighbourhood back from Python each tick
for [the visualizer](pose-manifold-embedding.md).

Two decisions worth keeping:

- **The neighbourhood is taken from Python, not recomputed in C#**, so the highlight shows the
  decision that was actually made. A second k-NN on this side could disagree with the one that
  produced the pose.
- **Bulk arrays are exposed as methods, not properties.** They run to thousands of entries, and
  anything that walks the component's properties by reflection — a debug inspector, a scene
  serializer, an editor bridge — would stall or dump megabytes if they looked like cheap getters.

A static delegate forwards Python's log output to Unity, because Python logs to stdout and PythonNET
does not forward that anywhere Unity shows. It is static so it stays rooted for as long as Python
might hold it.

## Known gaps

- **A mid-run exception disables the stage** to stop it spamming the console every frame. After that
  it silently stops updating the pose — while still returning `true`, so downstream stages keep
  running against a frozen pose.
- After that same failure, `Dispose` early-returns and the Python handles are never released under
  the GIL.
- **Foot contacts are captured once at `Init`** from the start state and never updated, so the contact
  booleans reported every frame derive from the start pose rather than the current one.
- **A bone-count disagreement across the boundary fails asymmetrically.** The unpacking loop is
  bounded by the *pose buffer's* bone count while indexing the flat Python arrays. So a Python side
  returning **fewer** bones than the rig reads past the end of those arrays and throws — caught by
  the handler above, which disables the stage and silently freezes the pose. A Python side returning
  **more** has the surplus silently ignored. Neither case names the two counts that disagreed, unlike
  [the FK path](../animation-tools/pose-buffers.md), which throws with both.
- **The stage never sets `PoseDiscontinuity`**, although `Playback` and `NearestNeighbour` replace the
  pose discontinuously by snapping to a database frame. The base class documents that as mandatory,
  so [inertialization](../motion-matching/inertialization.md) will not smooth those jumps.
- It checks the trained-ness flag but never the pose-database flag, so a stale `.mmpose` is not
  detected at runtime.

## Source map

| Concern | File |
| --- | --- |
| The stage | `Assets/MotionField/MotionFieldStage.cs` |
| Control-input base and the two concretes | `Assets/MotionField/MotionFieldControlInput.cs`, `MotionFieldDirectionControlInput.cs`, `MotionFieldSplineControlInput.cs` |
| Bone-weight marshalling | `Assets/MotionField/MotionFieldBoneWeights.cs` |

`MotionFieldBoneWeights` is shared by the runtime stage and both editor buttons deliberately: a value
function trained under one weight table and run under another is exactly what the staleness gate
exists to catch, and it can only catch it if both sides send the same thing.

**Tests.** None.

The Python side of each call is documented in
[the Unity call surface](../python/unity-call-surface.md) and
[motion field policies](../python/motion-field-policies.md).
