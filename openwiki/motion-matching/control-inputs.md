---
type: Architecture Guide
title: Control inputs
description: Turning intent into the trajectory the search matches against, the simulation-object model every input shares, and an honest account of which inputs actually work.
tags: [control-input, trajectory, crowd, status]
sources:
  - id: openwiki-source-08ea4bc02364c8785bdd8d8f
    resource: repo://Assets/AnimationTools/Runtime/Core/MotionSynthesisComponent.cs
  - id: openwiki-source-1fb0416756719f48922c424e
    resource: repo://Assets/MotionField/MotionFieldSplineControlInput.cs
  - id: openwiki-source-13742752b942a8c72fc71381
    resource: repo://Assets/MotionField/MotionFieldStage.cs
  - id: openwiki-source-eff87e6d0ba384d583d754cf
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/CollisionsSpringControlInput.cs
  - id: openwiki-source-27c6cae65749148a2fa4147b
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/CrowdControlInput.cs
  - id: openwiki-source-026c5ca3277e1544b8b26bd1
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/CrowdSplineControlInput.cs
  - id: openwiki-source-fd76ee323a7bc3a268f0ac68
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/DirectionControlInput.cs
  - id: openwiki-source-fed6ec6af0a6135c6cbeddce
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/MotionMatchingControlInput.cs
  - id: openwiki-source-5a2c119cb47a6d7fc76496cc
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/SplineControlInput.cs
  - id: openwiki-source-cd81b693f918262549b5d6bc
    resource: repo://Assets/Scripts/UserInput.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
---

# Control inputs

`MotionMatchingControlInput` is the *"what should the character be doing"* half of motion matching.
[The stage](matching-stage.md) is the *"which animation frame looks most like that"* half.

## The simulation-object model

Every subclass works the same way:

1. Drive a lightweight **simulation object** — usually the input's own Transform.
2. Predict where that object will be at each of the database's prediction horizons.
3. **Those predictions are the trajectory.**

Nothing here poses the character. The character follows only because the search keeps picking frames
that move like the prediction. That indirection is the whole design, and it is why a control input
can be swapped without touching anything downstream.

Prediction horizons are counted in **database frames**, so converting one to seconds means
multiplying by the database's own frame time — not by `Time.deltaTime`. The trajectory is a
ground-plane path, so Y is dropped and each value is two floats.

A control input can also raise `OnHighInputChange` to force an immediate search rather than waiting
out the [search interval](matching-stage.md).

## What is actually in use

Only **`DirectionControlInput` and the spline inputs** are live. That is worth stating precisely,
because the failure modes of the others differ and the naive summary is wrong.

| Input | Status |
| --- | --- |
| `DirectionControlInput` | live — used in 3 scenes/prefabs |
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

> **A quieter defect than the exceptions.** The four `Root*` properties on
> `MotionSynthesisComponent` — `RootPosition`, `RootVelocity`, `RootRotation`, `RootAngularVelocity`
> — are **never assigned anywhere in the repository**. They sit at `default`. So every crowd input
> that reads `RootPosition` to decide whether the character has fallen behind, or where to steer
> from, is silently measuring against the **world origin**, and `RootRotation` is
> `quaternion(0,0,0,0)` — not even identity. Nothing fails; the numbers are just wrong.

The whole obstacle-feature path is dead in the same way: `IObstacleAwareCharacterControler` has two
implementers and no consumer, and `Obstacle.EllipsesFeatureName` is referenced only from there.

See [the unimplemented block](../animation-tools/synthesis-pipeline.md) for what finishing this would
mean.

## DirectionControlInput

Stick and WASD control, and the simplest example of the model. A spring moves the Transform toward
the velocity the stick asks for, smoothing jerky input into something a body could do. **Running that
same spring further ahead, with no new input, is what produces the predicted trajectory.**

One asymmetry in the prediction worth knowing:

- **Facing predictions can be jumped** straight to each horizon in one step, because the implicit
  spring form is exact at any step size.
- **Position predictions must be chained** — each horizon continues from the previous one, because
  that spring carries acceleration and cannot be jumped.

It runs two independent springs, one for position and one for facing, precisely because facing and
travel are different quantities — see
[the simulation frame](../animation-tools/simulation-frame.md).

The reconciliation block — pulling the synthesized character back toward the simulation object, which
the search alone does not keep in sync — is present but **not wired**: all three methods are private
and uncalled, and the one call in `OnUpdate` is commented out. The intent is preserved in the
parameters, which cap the per-frame correction to a fraction of the character's own travel *so it
never outruns the animation and looks like sliding*.

Player input arrives through `AnimationTools.UserInput`, a singleton that wraps the generated
`InputActions` asset and re-broadcasts `Player.Move` as a serialized `UnityEvent<Vector2>`. Nothing
wires it in code — the hookup is made in the Inspector.

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
| Stick/WASD | `.../DirectionControlInput.cs` |
| Path following | `.../SplineControlInput.cs`, `.../PathControlInput.cs` |
| Crowd and collision | `.../CrowdControlInput.cs`, `.../CrowdSplineControlInput.cs`, `.../CollisionsSpringControlInput.cs` |
| Obstacles | `Assets/MotionMatching/Runtime/Unity/Obstacle.cs`, `ObstacleManager.cs` |
| Springs | `Assets/MotionMatching/Runtime/Utils/Spring.cs` |
| Player input | `Assets/Scripts/UserInput.cs`, `Assets/InputActions.cs` |

**Tests.** None.

**Ordering note.** Both the control inputs and `MotionSynthesisComponent` run in `LateUpdate` and
neither declares a `DefaultExecutionOrder`, so the order of "advance the simulation object" versus
"run the synthesis tick" is Unity's default component order rather than something pinned.
