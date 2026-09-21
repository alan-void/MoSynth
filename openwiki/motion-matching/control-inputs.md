---
type: Architecture Guide
title: Control inputs
description: Turning intent into the trajectory the search matches against, the simulation-object model every input shares, and an honest account of which inputs actually work.
tags: [control-input, trajectory, crowd, status]
sources:
  - id: openwiki-source-c47354627dcbea1099ddbb20
    resource: repo://Assets/AnimationTools/Runtime/AnimationTools.asmdef
  - id: openwiki-source-20e38e100e36c5379b2eb8ec
    resource: repo://Assets/AnimationTools/Runtime/ControlInput/IFrameTarget.cs
  - id: openwiki-source-a6b9fae1157f771557413c57
    resource: repo://Assets/AnimationTools/Runtime/ControlInput/MotionSynthesisControlInput.cs
  - id: openwiki-source-b106a4622123233262d982ae
    resource: repo://Assets/AnimationTools/Runtime/ControlInput/TrajectorySteering.cs
  - id: openwiki-source-21b6f7706116d71ad97ea47a
    resource: repo://Assets/AnimationTools/Runtime/ControlInput/UserInput.cs
  - id: openwiki-source-08ea4bc02364c8785bdd8d8f
    resource: repo://Assets/AnimationTools/Runtime/Core/MotionSynthesisComponent.cs
  - id: openwiki-source-1fb0416756719f48922c424e
    resource: repo://Assets/MotionField/MotionFieldSplineControlInput.cs
  - id: openwiki-source-13742752b942a8c72fc71381
    resource: repo://Assets/MotionField/MotionFieldStage.cs
  - id: openwiki-source-04495c5987b8f45114d94957
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/AnchoredDirectionControlInput.cs
  - id: openwiki-source-eff87e6d0ba384d583d754cf
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/CollisionsSpringControlInput.cs
  - id: openwiki-source-27c6cae65749148a2fa4147b
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/CrowdControlInput.cs
  - id: openwiki-source-fd76ee323a7bc3a268f0ac68
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/DirectionControlInput.cs
  - id: openwiki-source-fed6ec6af0a6135c6cbeddce
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/MotionMatchingControlInput.cs
  - id: openwiki-source-5a2c119cb47a6d7fc76496cc
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/SplineControlInput.cs
generated: {by: "claude-code", at: "2026-09-07T19:31:22.105Z"}
---

# Control inputs

`MotionMatchingControlInput` is the *"what should the character be doing"* half of motion matching.
[The stage](matching-stage.md) is the *"which animation frame looks most like that"* half.

## The model

Every subclass predicts where the character should be at each of the database's prediction horizons,
and **those predictions are the trajectory**. Nothing here poses the character; it follows only
because the search keeps picking frames that move like the prediction. That indirection is the whole
design, and it is why a control input can be swapped without touching anything downstream.

What differs between inputs is **where the prediction starts**, and the difference is not cosmetic:

- **From a simulation object the input drives itself** — usually its own Transform. The original
  model, and still what `DirectionControlInput` and the spline inputs do.
- **From the character's own frame, re-read every tick.** `AnchoredDirectionControlInput`.

### Why the origin matters

A simulation object and the synthesized character are never quite in the same place, and nothing
closes the loop between them. The difference accumulates and enters the query as a standing offset:
"you are behind where you should be", every tick. The character reads as permanently catching up.
That is the behaviour of the design, not a bug in the springs.

Anchoring the origin on the character removes the drift by construction — there is no position left
to drift. The trade is that such an input describes *motion* and never *location*, so it cannot place
a character or measure path-following error, which is why the spline inputs stay open-loop.

The third option is to stop asking the search to close the gap and place the character outright: see
[root following](../animation-tools/root-following.md), which is a stage rather than an input.

### Ordering

Control inputs advance in `Update`; `MotionSynthesisComponent` ticks in `LateUpdate`. So a search
always runs against this frame's trajectory rather than the previous frame's.

Prediction horizons are counted in **database frames**, so converting one to seconds means
multiplying by the database's own frame time — not by `Time.deltaTime`. The trajectory is a
ground-plane path, so Y is dropped and each value is two floats.

A horizon may be **negative**, meaning a point in the character's past
([why](feature-vectors.md)). The two live inputs answer that from opposite directions, and the
difference is not cosmetic:

- A **path follower** needs nothing extra. The path behind its current point *is* where it has
  been, so the same spline evaluation with a negative frame count answers it.
- A **spring-driven input cannot compute its own past.** The spring says where the object is going,
  and stepping it with a negative timestep does not run it backwards. `DirectionControlInput`
  therefore records a `TrajectoryHistory` — a ring of timestamped ground-plane samples, read back
  by interpolation — and answers past horizons from that.

A control input can also latch a *high input change* flag to force an immediate search rather than
waiting out the [search interval](matching-stage.md); the stage consumes it each tick.

## How a control input finds its character, and how the player finds it

Every control input — motion matching, PFNN and motion field alike — derives from
`AnimationTools.MotionSynthesisControlInput`. That base holds **the only reference between an input
and the character it steers**: a serialized `MotionSynthesisComponent`, falling back to
`GetComponentInParent` when the field is left empty, so an input dropped on a character prefab is
already wired.

The link is one-directional on purpose. Nothing downstream carries a reference back: the input
*claims* the character in `OnEnable`, and a stage that needs it reads
`MotionSynthesisComponent.ControlInput` and casts. Pairing two serialized fields by hand was the
older shape, and the two could disagree — a scene could point a stage at one input while a second
input pointed at the same character.

A character takes one input at a time. A second input enabling on a claimed character is warned
about by name and disables itself, so choosing between two inputs on one character is done by
deactivating the one you do not want, which is what the demo scene does. An `OnEnable` override that
acquires anything of its own must call base and then check `IsBound`, or it will set up state on a
component that has just switched itself off.

Player input needs no wiring either. `UserInput` publishes the move action on a **static**
`MoveChanged` event, and the base subscribes any subclass implementing
`IMotionSynthesisDirectionControlInput` for as long as it is enabled, seeding it with the current
stick value so an input that enables mid-session does not start from zero. Static rather than a
reference each listener is handed, because a control input arrives with its character — dropped in as
a prefab, or spawned by the benchmark sweep — and has no way to reach a scene object it was never
wired to.

This is why `UserInput` and `InputActions.inputactions` live under `Assets/AnimationTools/`, and why
`AnimationTools.asmdef` references `Unity.InputSystem`: an asmdef assembly cannot reference
`Assembly-CSharp`, where `UserInput` used to sit, so subscribing to it directly was impossible until
it moved.

## What is actually in use

Only **`DirectionControlInput` and the spline inputs** are live. That is worth stating precisely,
because the failure modes of the others differ and the naive summary is wrong.

| Input | Status |
| --- | --- |
| `DirectionControlInput` | live — used in 3 scenes/prefabs |
| `AnchoredDirectionControlInput` | live — the anchored-origin variant of the above |
| `SplineControlInput` | live — 2 |
| `SplinePoseKeypointControlInput` | live — 1, and actively being worked on |
| `CrowdControlInput` | **throws on its first update** with default inspector values |
| `CollisionsSpringControlInput` | **throws on its first update** with default inspector values |
| `CrowdSplineControlInput` | does *not* throw, but is unused and reads zeros (below) |
| `PathControlInput` | calls none of the unimplemented API; simply unused |
| `MoveForwardInput` | a 19-line test harness |

The two that throw do so because they call `SetPosAdjustment` / `SetRotAdjustment`, which throw
`NotImplementedException` — and both `DoClamping` and `DoAdjustment` default to `true`, so the path
runs immediately.

`CrowdSplineControlInput`'s only throwing call sits inside `GetNearbyObstacles`, which **has no call
sites anywhere**. So it would run. `PathControlInput` calls none of the four unimplemented members at
all.

> **A defect that has since been fixed.** The four `Root*` properties on `MotionSynthesisComponent`
> used to be unassigned auto-properties sitting at `default`, so every crowd input reading
> `RootPosition` to decide whether the character had fallen behind was silently measuring against the
> **world origin**. Nothing failed; the numbers were just wrong. They now route to the component's
> Transform and to rates re-derived from the pose each tick.

The whole obstacle-feature path is dead in the same way: `IObstacleAwareCharacterControler` has two
implementers and no consumer, and `Obstacle.EllipsesFeatureName` is referenced only from there.

See [the unimplemented block](../animation-tools/synthesis-pipeline.md) for what finishing this would
mean.

## AnchoredDirectionControlInput

The same springs as `DirectionControlInput`, with the origin re-read off the character every tick and
no position state of its own. Two decisions in it are worth knowing, both about *not* keeping state:

- **The velocity springs keep their own state rather than being seeded from the character's measured
  velocity.** Seeding looks more honest and is worse: the rate read back off the rig carries the
  hips' sway and the clip's noise, which would enter the query directly and then influence the search
  that produced it.
- **There is no facing state at all.** Each horizon's facing is damped from the character's actual
  yaw toward its direction of travel. The database bakes Direction channels as *facing*, and the
  character's facing is whatever yaw rate the matched clip integrated, so a stored facing would be a
  second source of truth the clip can contradict — the same defect as the free simulation object, in
  smaller form.

Negative horizons still come from a `TrajectoryHistory`, recording where the character has been
rather than where a simulation object went.

It shares its steering maths with the PFNN direction input through `TrajectorySteering` in
`AnimationTools`: rate-limit the request, run the velocity spring on, jump the facing damper to each
horizon in closed form.

## DirectionControlInput

Stick and WASD control, and the simplest example of the model. A spring moves the Transform toward
the velocity the stick asks for, smoothing jerky input into something a body could do. **Running that
same spring further ahead, with no new input, is what produces the predicted trajectory.**

One asymmetry in the prediction worth knowing:

- **Facing predictions can be jumped** straight to each horizon in one step, because the implicit
  spring form is exact at any step size.
- **Position predictions must be chained** — each horizon continues from the previous one, because
  that spring carries acceleration and cannot be jumped. Horizons must therefore be authored in
  ascending order, since the chain steps by their differences.

Negative horizons are skipped by both springs and answered from `TrajectoryHistory` instead. Before
the history reaches that far back — the first second or so of a run — the query falls back to the
object's current state, which is what a character that had been standing still would have recorded
anyway.

It runs two independent springs, one for position and one for facing, precisely because facing and
travel are different quantities — see
[the simulation frame](../animation-tools/simulation-frame.md).

The reconciliation block — pulling the synthesized character back toward the simulation object, which
the search alone does not keep in sync — is present but **not wired**: all three methods are private
and uncalled, and the one call in `OnUpdate` is commented out. The intent is preserved in the
parameters, which cap the per-frame correction to a fraction of the character's own travel *so it
never outruns the animation and looks like sliding*.

Player input arrives through `AnimationTools.UserInput`, a singleton that wraps the generated
`InputActions` asset and publishes `Player.Move` on a static event every direction input subscribes
to itself. The serialized `UnityEvent<Vector2>` beside it remains for listeners that are not control
inputs.

## SplineControlInput

Follows a spline at constant speed, looping. The trajectory is read straight off the spline rather
than simulated.

**Nothing here reacts to where the character actually is.** The point on the spline advances on its
own clock, so drift shows up as measurable path-following error instead of being steered out. That is
exactly what makes it a **measurement tool** rather than a gameplay controller, and it is why
[`PathFollowingMetric`](../animation-tools/path-following-metrics.md) drives it.

Two implementation details carry real arguments:

- **`HasPath` is checked before evaluation, not inside it.** A container whose spline list has been
  emptied *throws* from `EvaluatePosition` and `CalculateLength` rather than answering null.
- **A degenerate path length holds position.** When the path length is at or below a small epsilon,
  `OnUpdate` returns without advancing and the character simply stays where it is. Without that
  guard the parameter advance would be infinite, `frac` of infinity is NaN, and the NaN would stick
  in the follower's state and leave as a non-finite character position.
- **One direction seam.** Every direction goes through a single method, so an override reaches the
  trajectory query, the debug gizmos and `GetCurrentRotation` together. A bare path carries no facing
  of its own, so that seam measures **direction of travel** — as a short step further along rather
  than as the analytic tangent, which keeps it consistent with the predicted positions.

For spawn facing it delegates to
[`SplinePathDirection`](../animation-tools/simulation-frame.md), because a pure path follower has no
notion of facing beyond its path and the Transform's own forward is arbitrary for a character spawned
onto a path it was never authored against.

A path that *does* carry facing overrides that seam — see
[spline pose keypoints](spline-pose-keypoints.md).

## The crowd and collision family

These are unused, but the reasoning in them is worth keeping.

**Two mechanisms, easily confused.** *Steering* deflects the simulation object so the character walks
around neighbours. *Obstacle features* put nearby obstacles into the query vector so the search can
prefer animations recorded while avoiding something. Either works alone; together the character both
moves around a neighbour and looks like it meant to.

**The steering algorithm.** A fan of rays is cast ahead, and **only the closest hit steers** —
avoiding several at once averages into steering nowhere. The force is perpendicular to forward, so it
sidesteps rather than brakes, and a log falloff makes distant obstacles barely matter while close
ones dominate.

**Which side to pass on is a coordination problem.** If both characters dodge the same way they still
collide. So when the obstacle is itself steering, the avoidance takes the **opposite** side. Static
obstacles are skipped entirely — walls do not negotiate.

**`CrowdSplineControlInput`** adds avoidance to a fixed path, and can hold the point on the spline
back when the character falls behind so avoidance does not let the target run away. It notes its own
consequence honestly: that also makes it **a poorer path-following benchmark, since the reference
reacts to the character it is measuring**.

**`CollisionsSpringControlInput`** sweeps the simulation object against colliders and drops it onto
the floor. Both raycasts act on the **simulation object, never on the synthesized character** — so
the trajectory handed to the search already respects the environment, and the search picks an
animation that turns *along* a wall rather than one that walks into it. The wall ray starts a step
*behind* the current position, so an object already slightly inside geometry still sees the surface;
surfaces facing more up than sideways are treated as floors instead.

**`Obstacle`** models something to walk around as an upright cylinder — a ground circle plus a height
band — which is all the avoidance maths needs and is cheap to raycast. A non-static obstacle expects
to be a sibling of a crowd input under a shared parent, and *that link is what lets two characters
agree which way to pass each other*. `ObstacleManager` runs at execution order −1000 so obstacles can
register before inputs look for them, and announces membership changes once per frame so a batch of
spawns costs subscribers one rebuild.

## Springs

`Spring` is the shared implicit-damper family, credited to Daniel Holden's *spring roll call*. The
implicit forms are unconditionally stable. Rotational springs go through scaled-angle-axis with
shortest-arc correction — see
[the rotation-rate rule](../animation-tools/channel-layout-system.md).

## Source map

| Concern | File |
| --- | --- |
| Base class and the model | `Assets/MotionMatching/Runtime/CharacterController/MotionMatchingControlInput.cs` |
| Stick/WASD | `.../DirectionControlInput.cs`, `.../AnchoredDirectionControlInput.cs` |
| Shared steering maths | `Assets/AnimationTools/Runtime/ControlInput/TrajectorySteering.cs` |
| Path following | `.../SplineControlInput.cs`, `.../PathControlInput.cs` |
| Crowd and collision | `.../CrowdControlInput.cs`, `.../CrowdSplineControlInput.cs`, `.../CollisionsSpringControlInput.cs` |
| Obstacles | `Assets/MotionMatching/Runtime/Unity/Obstacle.cs`, `ObstacleManager.cs` |
| Springs | `Assets/AnimationTools/Runtime/Utils/Spring.cs` |
| Shared base: the character reference, the claim, the input subscription | `Assets/AnimationTools/Runtime/ControlInput/MotionSynthesisControlInput.cs` |
| Player input | `Assets/AnimationTools/Runtime/ControlInput/UserInput.cs`, `.../InputActions.cs` |

**Tests.** `ControlInputPlanarHelpersTests` pins the character-frame conversion every input answers
the query through. `TrajectoryHistoryTests` covers the past-horizon ring, and
`SplineControlInputFoldTests` the open-path clamp.
