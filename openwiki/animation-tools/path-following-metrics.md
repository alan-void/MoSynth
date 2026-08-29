---
type: Architecture Guide
title: Path following and its metrics
description: Windowed spline projection and why global nearest-point is wrong, plus how trajectory, heading, speed and lap progress are measured.
tags: [spline, projection, metrics, evaluation]
sources:
  - id: openwiki-source-ef6c8dadf2b618e1ca697c96
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/SplineLapTracker.cs
  - id: openwiki-source-531380767041ea3e07c039ed
    resource: repo://Assets/AnimationTools/Runtime/Evaluation/PathFollowingMetricsCalculator.cs
  - id: openwiki-source-102332f0c8186b273774e522
    resource: repo://Assets/AnimationTools/Runtime/Evaluation/SplineProjector.cs
  - id: openwiki-source-d839be2803f7ce43ccafb7ca
    resource: repo://Assets/AnimationTools/Tests/Editor/PathFollowingMetricsCalculatorTests.cs
  - id: openwiki-source-f119139da5246c18d0310100
    resource: repo://Assets/AnimationTools/Tests/Editor/SplineLapTrackerTests.cs
  - id: openwiki-source-a7e5f5ed2b97c27dd1d6c658
    resource: repo://Assets/AnimationTools/Tests/Editor/SplineProjectorTests.cs
  - id: openwiki-source-1fb0416756719f48922c424e
    resource: repo://Assets/MotionField/MotionFieldSplineControlInput.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Path following and its metrics

Measuring how well a character follows a path needs one non-obvious primitive — a projection onto the
spline that **preserves continuity** — and three metrics built on it.

| Piece | Role |
| --- | --- |
| `SplineProjector` | stateful, continuity-preserving point-to-spline projection |
| `PathFollowingMetricsCalculator` | pure; recorded positions and forwards → trajectory, heading and velocity error |
| `SplineLapTracker` | a stream of normalized parameters → how far round the path the character got |
| `PathFollowingMetric` | the in-scene harness that runs the above over live characters |

Everything here is XZ-flattened, and the spline container is assumed **unscaled** — scaling it would
skew the arc-length maths.

## Why global nearest-point is wrong

This is the argument the projector exists for.

`SplineUtility.GetNearestPoint` searches the whole spline. On a path that crosses itself that is
**actively wrong**: at the crossing, the globally nearest point flips to the other branch, and every
consumer of the result flips with it —

- a pure-pursuit controller steers onto the wrong lobe,
- a lap counter reads the flip as a seam crossing,
- an error metric compares the character against a tangent pointing the other way.

The Unity Splines package offers no windowed alternative. Both public `GetNearestPoint` overloads
hardcode a full-spline range; the range-aware internals are private; and `SplineSlice` is limited to
whole knots and documents itself as unsuited to per-frame use. So `SplineProjector` mirrors the
package's own coarse-to-fine chord search over a caller-controlled window.

Expressing that window in **metres** is possible because normalized spline parameters are uniform in
arc length: a window of *d* metres is `d / spline.GetLength()` in normalized space, on any spline,
anywhere along it.

### How the window is sized

The window is derived from the point's own displacement since the last query, not fixed:

```
forwardMeters = max(minWindowMeters,        displacement * windowSlack)
backMeters    = max(minWindowMeters * 0.5f, displacement * 0.5f)
```

Asymmetric — it looks further ahead than behind. Sizing from actual movement is what lets one
projector serve both a 30 Hz controller stepping centimetres and a recording sampled seconds apart.
`SplineProjectorTests` covers exactly that: 10 m steps on a 40 m perimeter still land within 0.2 m.

Defaults: `minWindowMeters = 0.5`, `windowSlack = 3`, `reacquireDistanceMeters = 5`. Each query runs
`RefinementPasses = 3` coarse-to-fine passes, with a floor of `MinimumSegments = 8` chords per pass —
below that the search can step over a fold in the path. Each pass narrows to two chords either side of
the winner, so the true minimum cannot fall outside the next window.

Open splines clip the window at the ends; closed splines wrap.

### Re-acquisition is deliberately reluctant

Past `reacquireDistanceMeters` from the windowed result, the projector gives up on continuity and
searches globally. That threshold is generous on purpose: **re-acquiring is precisely what must not
happen at a crossing**, and a crossing is close to both branches, so it never trips. Only a point
that has genuinely lost the path gets to jump.

`Reset()` forgets the anchor so the next query seeds globally — call it whenever the point stops
being the same point: a different spline, a respawned character, a new benchmark run.

## The metrics

`PathFollowingMetricsCalculator.Evaluate` produces trajectory error, heading error and velocity
error, all over the frames after `settleTime` (start-up transients are dropped).

It walks **one projector forward** across those frames rather than re-deriving per frame. Doing it
globally would flip branches at a crossing and manufacture a roughly **180 degree** heading error out
of a run that never left the path — and there is a test asserting a perfect run on a self-intersecting
spline scores near zero on both terms.

Details worth keeping:

- **Heading error uses an `atan2` form**, not `acos(dot)`, because `acos` loses precision for
  nearly-parallel directions.
- **Actual speed is a finite difference of the recorded root.** Consecutive analysis frames are
  differenced in the ground plane and divided by the recorded time delta, skipping any pair whose
  delta is not positive. Note the smoothing window is then sized from the **mean of those deltas**,
  not from the configured synthesis frame rate — so a recording sampled per Unity frame and one
  sampled per synthesis tick get appropriately different windows.
- **Speed is smoothed before velocity error.** Per-tick speed is dominated by synthesis jitter —
  frame switches, inertialization, gait-cycle sway — so the raw signal inflates `|actual − target|`.
  A window of about **0.2 s** keeps gait-level changes visible while suppressing that.
- The average is **centred, not trailing**, so smoothed speed stays time-aligned with the frames it
  came from instead of lagging by half a window. Near the ends the window is truncated to what exists
  rather than padded or dropped.
- Fewer than two analysis frames yields `framesEvaluated == 0` and every metric NaN.
- A NaN target speed propagates: trajectory and actual speed stay finite, velocity error is NaN.

One asymmetry that is intentional but easy to misread: mean trajectory error divides by the analysis
frame count, while mean heading error divides by the count of frames that had a *usable tangent*.

## Lap tracking

`SplineLapTracker` turns a stream of normalized parameters into progress.

**Why progress rather than a timer.** A guess is not available: `MotionFieldSplineControlInput`
reports `float.NaN` for target speed because its policy has no speed model, so lap time cannot be
derived from path length. Measuring the character's own progress also keeps the comparison honest —
a method that falls behind gets a *longer run* rather than a truncated lap.

For a closed spline, progress accumulates **signed deltas**, each wrapped into ±half a lap. A single
tick never covers half the spline, so a delta larger than that is the seam being crossed rather than
the character teleporting. This is also what makes the tracker robust to the backward jitter a
synthesized root produces every gait cycle: it subtracts rather than counting as a lap in the other
direction. Progress can therefore go negative if the character runs backwards, and it never completes.

Open splines cannot lap, so progress is just the current parameter and completion means reaching
`OpenSplineEndT = 0.995f` — short of 1 because the nearest point to a character that has run past the
end clamps to the final knot, which a strict test would still reject on floating point.

`Reset()` drops accumulated progress and re-anchors, and is called once the settle time elapses so
the measured lap covers settled motion rather than the character's scramble up to speed.

## Running it standalone

`PathFollowingMetric` is a MonoBehaviour harness. Assign a spline, list the spline control inputs,
press Play. It pushes the spline into each input, spins up a `MotionRecorder` per input, runs for
`duration` seconds, reads the recordings back, and logs a block per character:

```
[PathFollowingMetric] <label>: <n> frames evaluated (settle 2.0 s)
  Trajectory error   mean 0.000 m
  Heading error      mean 0.0 deg, max 0.0 deg
  Speed              actual 0.00 m/s, target 0.00 m/s, mean abs error 0.00 m/s
```

The last line reads `target n/a (no speed model)` when the target speed is NaN. Each entry is
individually try/caught, so one bad recording does not abort the rest. `duration = 0` means manual
only. Defaults: 20 s duration, 2 s settle, output to `Recordings/`.

The same calculator runs inside a sweep via `BenchmarkRunEvaluator` — see
[benchmarking](benchmarking.md).

Note that the harness adds every recorder to **its own** GameObject rather than to each character, so
N recorders coexist on the metric object, each pointed at a different synthesizer.

## Source map

| Concern | File |
| --- | --- |
| Windowed projection | `Assets/AnimationTools/Runtime/Evaluation/SplineProjector.cs` |
| The three errors | `Assets/AnimationTools/Runtime/Evaluation/PathFollowingMetricsCalculator.cs` |
| In-scene harness | `Assets/AnimationTools/Runtime/Evaluation/PathFollowingMetric.cs` |
| Lap progress | `Assets/AnimationTools/Runtime/Benchmark/SplineLapTracker.cs` |

**Tests.** `SplineProjectorTests` is unusually well-constructed: alongside the straight-line agreement
case and monotonic perimeter tracking, it has both `FigureEightDoesNotJumpBranchesAtTheCrossing` —
the regression the class exists for — and its **negative control**,
`FigureEightGlobalSearchDoesJumpBranches`, which proves the fixture actually exercises the bug rather
than being a path that was never ambiguous.

`PathFollowingMetricsCalculatorTests` covers each error term against hand-computed values, the NaN
propagation contract, settle-time exclusion, and the self-intersecting case.
`SplineLapTrackerTests` covers seam crossing, multi-lap counting, backward jitter, negative progress
and the open-spline branch.
