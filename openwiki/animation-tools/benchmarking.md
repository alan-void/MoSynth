---
type: Operations Guide
title: Benchmarking synthesis methods
description: The Editor-driven sweep that runs every method against every path, what it measures, and what those numbers may and may not be compared against.
tags: [benchmark, metrics, operations, editor]
sources:
  - id: openwiki-source-7307d1c181463c9f2142b2af
    resource: repo://Assets/AnimationTools/Editor/Benchmark/BenchmarkPathGenerator.cs
  - id: openwiki-source-78ea72b8000389e96e8b6d0f
    resource: repo://Assets/AnimationTools/Editor/Benchmark/BenchmarkRunEvaluator.cs
  - id: openwiki-source-9aee3e250e99cf3e0effe33d
    resource: repo://Assets/AnimationTools/Editor/Benchmark/RandomBenchmarkPathGenerator.cs
  - id: openwiki-source-089bfe06327e84b6a0f8164a
    resource: repo://Assets/AnimationTools/Editor/Benchmark/RandomPathShapes.cs
  - id: openwiki-source-570b283522112de5884130ea
    resource: repo://Assets/AnimationTools/Editor/Benchmark/SynthesisBenchmarkCli.cs
  - id: openwiki-source-fdcd1ba2cc82f452d31d0aa6
    resource: repo://Assets/AnimationTools/Editor/Benchmark/SynthesisBenchmarkDriver.cs
  - id: openwiki-source-965f4951924717adcf6f5f36
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/BenchmarkLapProbe.cs
  - id: openwiki-source-e67ed4d7db4d26fc3b903c56
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/BenchmarkOverride.cs
  - id: openwiki-source-d0f9d43aec8111ff411d1e4d
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/MetricsMath.cs
  - id: openwiki-source-721e7f001dd1252e3e9fe0e4
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/MotionQualityMetricsCalculator.cs
  - id: openwiki-source-d7098198afbab4867439f2bc
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/MotionQualityMetricsResult.cs
  - id: openwiki-source-ef6c8dadf2b618e1ca697c96
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/SplineLapTracker.cs
  - id: openwiki-source-ee25991e1d2c2f075a2ba7f1
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/SynthesisCostMetricsResult.cs
  - id: openwiki-source-102332f0c8186b273774e522
    resource: repo://Assets/AnimationTools/Runtime/Evaluation/SplineProjector.cs
  - id: openwiki-source-bcdc84423195ac552312e1fe
    resource: repo://Assets/AnimationTools/Runtime/Utils/SplineFold.cs
  - id: openwiki-source-1c4feddebb99d6bb6efffd0f
    resource: repo://Assets/AnimationTools/Tests/Editor/MotionQualityMetricsCalculatorTests.cs
  - id: openwiki-source-eb8c72ed20ac6079d2988204
    resource: repo://Assets/MotionField/Benchmark/MotionFieldBenchmarkOverrides.cs
  - id: openwiki-source-8331c1b2b04b3cc432256c64
    resource: repo://Assets/MotionMatching/Runtime/Benchmark/MotionMatchingBenchmarkOverrides.cs
  - id: openwiki-source-5a2c119cb47a6d7fc76496cc
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/SplineControlInput.cs
  - id: openwiki-source-0005a5fab5e50ca368e2a4f7
    resource: repo://Tools/run-benchmark.ps1
generated: {by: "claude-code", at: "2026-08-31T18:45:35.579Z"}
---

# Benchmarking synthesis methods

The sweep runs every configured method against every path, one character alive at a time, and writes
a report. It is an **Editor-side state machine plus a single runtime probe**.

> **Benchmark numbers are not a design constraint.** Synthesis methods here are not finalised. These
> figures exist to compare methods against each other, not to be defended as targets.

## Why the driver lives in the Editor

Two stated reasons: loading path and character prefabs needs `AssetDatabase`, which has no business
in the runtime assembly; and driving the sweep from outside the scene means **no state has to survive
between runs**.

So the queue is pumped from `EditorApplication.update`, and the only runtime piece is
`BenchmarkLapProbe`, which answers the one question the Editor cannot: has this character been round
the path yet.

**Runs never overlap.** Two characters sharing a frame would contend for the same cores and the same
Python GIL, and the cost metrics would then be measuring that contention rather than either method.

## The run lifecycle

```
WaitingForPlayMode → Spawn → Running → Teardown → Cooldown → (Spawn | Complete)
```

`StartRun` has a delicate ordering, and each step earns its place:

1. Instantiate the character **under an inactive holder**, so `Awake` does not run yet.
2. Find the synthesis component and the spline control input.
3. Assign the spline to the input — **before** placement, because the input cannot say which way to
   face on a path it has not been given.
4. Place the character at the path start.
5. Apply the method's overrides.
6. Reparent to the scene root, which activates the character and finally runs `Awake` — against the
   placed transform.

The inactive holder is the crux. `MotionSynthesisComponent` seeds its pose from the Transforms in
`Awake`, so a character that wakes at the origin and is moved afterwards **starts with a bogus root
velocity**. Overrides apply while the root is still inactive, so whatever they set is what `Awake`
sees.

Placement moves the root by whatever delta lands the *character frame* on the path — the synthesis
component's own transform, not the prefab root, since that is what root motion drives and what every
measurement reads. In these prefabs it sits about two metres off the root. Rotation is applied first,
then the translation is measured against where the frame ended up, so the character's internal
offsets survive.

Facing follows the [precedence rule](simulation-frame.md): the control input decides, and the path's
own start direction is only the fallback. A path carrying its own facing therefore spawns the
character strafing or backpedalling when that is what it was authored to do.

## Timing control

`Time.captureDeltaTime` decouples the run from the wall clock: every rendered frame advances exactly
one synthesis tick, so the sweep runs as fast as the CPU allows and produces **the same tick count
every time**.

Vsync is disabled alongside it, because with frames no longer tied to real time vsync is the only
thing left that would still pace them — and it would silently undo the fast-forward in a windowed
Editor. `Application.runInBackground` is set for the same class of reason: an unfocused Editor
otherwise throttles to a few ticks a second.

A finished character is destroyed with `DestroyImmediate` followed by an explicit `GC.Collect()`,
so that stage teardown — which for the motion field releases Python state — has finished before the
next run's cost is measured.

## Partial results are kept

A report is written whenever there is anything to write, **not only on success**. A sweep that was
cut short still produced real numbers for the runs that completed, and throwing those away helps
nobody. Success governs the exit code alone.

Each finished run is also appended to `progress.jsonl` as it completes, so a sweep that is killed
half way still leaves usable numbers behind.

The inverse rule applies to runs that did not happen: a run that lost its recorder or probe writes
**no row at all**, because recording an empty row would put a zero into the report as if it were a
measurement.

Play mode ending under the sweep takes the scene with it, so the driver stops at the first sign of
that and keeps the runs that did finish.

## Overrides

`BenchmarkOverride` lets one prefab cover several methods — the four motion field policies, or a
`searchInterval` ladder. Concrete overrides live next to the thing they poke, in the assembly that
depends on `AnimationTools`; see [the seam rule](synthesis-pipeline.md).

| Override | Assembly | What it changes |
| --- | --- | --- |
| `SynthesisFrameRateOverride` | AnimationTools | synthesis tick rate |
| `StageEnabledOverride` | AnimationTools | toggles a stage by type name |
| `SearchIntervalOverride` | MotionMatching | seconds between searches |
| `SplineSpeedOverride` | MotionMatching | the spline follower's constant pace |
| `MotionFieldPolicyOverride` | MotionField | which policy runs |
| `MotionFieldLookaheadOverride` | MotionField | pure-pursuit lookahead distance |
| `MotionFieldStartStateOverride` | MotionField | starting database state |

`SearchIntervalOverride` is **the headline quality/cost dial** for motion matching: searching every
tick tracks better and costs more, which is exactly what the sweep exists to quantify. It defaults to
ten database frames at 60 Hz.

`SplineSpeedOverride` has a coupling worth knowing: it changes the pace the follower travels at,
which is *also* the target the velocity error is measured against. It moves the run and the yardstick
together.

### The overrides that are deliberately absent

There are **no overrides for `MotionFieldConfig` hyperparameters** — `kNeighbors`, `tugRatio`, the
per-bone weights. Those live on a shared ScriptableObject, so writing them would dirty the asset and
leak into every later run in the sweep. Sweeping them needs a config asset per variant.

`StageEnabledOverride` matches by `GetType().Name` string with no namespace, which is
refactor-fragile by construction. It warns when nothing matched.

## What is measured

Everything is computed **from the recording on disk**, never from live state, so a report can be
rebuilt from the `.bin`/`.json` pair alone and the calculators stay pure and unit-testable. Channels
are looked up by name and treated as optional: a narrower recording still yields whichever metrics
its columns support, with the rest NaN. The one exception is the root channel — every quality figure
is anchored on the root's trajectory, so without it there is nothing to compute rather than something
to compute badly.

### Path following

Trajectory error, heading error and velocity error — see
[path following metrics](path-following-metrics.md).

### Motion quality

These exist to separate two methods that score *identically* on path following. A motion field
snapping between database neighbours and a matching search jumping to a distant frame both hit their
trajectory, and both look wrong; footskate and root jerk are what measure that wrongness.

| Metric | Meaning |
| --- | --- |
| `footskatePerMeter` | metres a foot slides while flagged planted, per metre the root travelled — scale-free, so it compares across speeds |
| `contactFraction` | fraction of frames with at least one foot in contact; a sanity check on the above |
| `rootJerkP95` | 95th-percentile root jerk, where single-frame pops show up rather than average sway |
| `discontinuitiesPerSecond` | ticks per second where a stage replaced the pose discontinuously |

**Missing contact data reports NaN, not zero.** No contact frame at all means the stage never wrote
foot contacts, not that the feet never slipped — reporting 0 would score a method with no contact
data as having perfect foot planting, which is exactly backwards. There is a test named for that.

Jerk needs four positions to produce one sample, so fewer than four analysis frames yields all NaN.

### Cost

Mean, median, 95th percentile and max of per-tick `Apply` time, plus GC bytes per tick and a
per-stage breakdown.

Percentiles are **nearest-rank rather than interpolated**, because these samples are per-tick costs
and jerks: the interesting figure is a value that actually occurred, not a blend of two.

Settle frames are dropped from the cost metrics as well as the quality metrics, because the first
ticks carry one-off setup — a motion field importing its Python modules, a search warming its caches
— that would otherwise dominate the mean.

> **What cost numbers can be compared against.** Wall clock inside a possibly-batchmode Editor is not
> player performance. It is valid for comparing methods against each other **on one machine in one
> sweep**, which is the only claim the benchmark makes. The JSON report records the machine and Unity
> version specifically so the numbers are not accidentally compared across either.

## The standard paths

Hand-drawn splines are fine for looking at a character but make a poor benchmark: nobody can say
afterwards what a method was actually being asked to do. The four generated paths are parametric and
regenerable, and each isolates a different demand.

| Path | Isolates |
| --- | --- |
| `Circle(4)` | constant curvature — the baseline everything else is worse than |
| `Oval(7, 3)` | straights joined by tight ends, separating acceleration into a turn from the turn |
| `FigureEight(6)` | the only path that **reverses turn direction** — where a policy committed to a turn shows whether it can commit to the opposite one |
| `SharpCorners(6, 4)` | linear tangents, so corners are genuinely discontinuous in heading; no locomotion clip can turn that fast, so corner overshoot dominates |

All four are closed, because a lap is the run length and an open path can only be traversed once. All
lie in the XZ plane with an unscaled container, which the metrics and the motion field's spline input
both assume.

## Random paths

Four curated shapes is a small sample, and a method can be tuned — deliberately or not — to the exact
set it is scored on. `MoSynth > Benchmark > Create Random Paths…` widens the suite without giving up
the property that makes it worth having: that you can say afterwards what the method was asked to do.

A path's geometry is a pure function of its kind, seed and slot, and the name carries the last two —
`Random_s4821_03_SharpWalk`. A `results.csv` row therefore identifies the exact curve, and the same
seed rebuilds it. The seed is sub-hashed **per slot** rather than drawn from one stream shared across
the batch, so changing the mix does not disturb the paths that keep their kind; with a shared stream
every prefab after the first change would silently differ.

Six families: three shapes, each drawn once with rounded turns and once with corners.

| Shape | Construction | Demand |
| --- | --- | --- |
| `Loop` | radius perturbed around a circle at sorted angles; closed | sustained turning, and a second lap to recover on |
| `Line` | an open corridor advancing along one axis with a lateral wobble | no repeated geometry to settle into |
| `Walk` | an open path that turns by a random amount at every knot | no repeated geometry *and* no global containment |

`Smooth` families are auto-smoothed; `Sharp` families use linear tangents, so the spline *is* its
knot polygon, and swing harder. A batch is partitioned by three nested ratios: smooth against sharp,
then within each, loops against the rest, then walks against the corridors — so the walk share is a
share of *what does not loop*, and each ratio stays independently meaningful.

**Two dials, and they mean different things.** The extent is how big a path is: a loop's radius, and
for the open shapes a per-knot step sized to give the same total length as a loop of that extent — so
runs stay comparable in duration whatever the shape. The knot-count range is separately how
*convoluted* a path of that size gets.

**Why the closed families are built in polar form.** Knots at strictly increasing angles with a
positive radius give a star-shaped — therefore simple — knot polygon, which is a proof rather than a
test for the linear-tangent family, where the spline *is* its polygon.

**Why open paths are corridors, not unclosed rings.** Dropping the closed flag on a ring leaves the
start and end one segment apart, so "reached the far end" becomes indistinguishable from "back at the
start" and the pure-pursuit lookahead near the end points across the gap. Advancing monotonically
along one axis also makes the polyline the graph of a function of that axis, so it cannot cross
itself at all.

**Why the walk is the awkward one.** Being contained by nothing is the point of it — a method that
has settled into "orbit" or "go roughly straight" is never asked for anything else by the other four
families. The cost is that it is the one shape with no construction-level argument against crossing
itself. It leans entirely on the rejection loop, and what makes those retries converge rather than
exhaust is that each relaxation shrinks its per-knot heading cap, straightening the walk.

**No minimum turn radius, on purpose.** An earlier version held the smooth families to one and sized
their wobble from it. It was dropped: the sharp families never had such a bound, and how well a
method follows a demanding curve is the measurement rather than a defect. The knot count is the dial
that decides how tight a path gets, and it is left free.

**Rejection, not hope.** Every candidate is redrawn until it stops crowding itself; the sharp
families must additionally clear a minimum corner separation and a per-shape turn-angle cap. A slot
that cannot be satisfied is skipped with an error naming *the predicate that rejected the last
candidate, with its measured value* — a user-settable knot range makes unsatisfiable settings easy to
ask for, and 20 knots inside a 5 m extent leaves the sharp families 0.5 m between corners, which is a
far more useful thing to be told than that generation failed. Over 200 seeds none of the six families
exhausts its attempts at the default settings.

**What replaced the turn-radius check.** Since nothing rejects a tight curve any more, the generation
log reports one per path: the tightest turn radius for a smooth family, and the sharpest corner angle
for a sharp one, where a circumradius would be meaningless — a linear-tangent corner is a genuine
curvature singularity. So a path that demands a 0.8 m turn still gets written, and you can see that
it does.

**Why the clearance test is not just a crossing test.** `SplineProjector` already survives a clean
crossing — `FigureEight` self-intersects and is a standard path. The case that breaks it is two
branches running close and parallel, where the windowed search can slide onto the wrong one and the
lap tracker reads the resulting parameter jump as a seam crossing, manufacturing a lap the character
never ran. So candidates are rejected for *approaching* themselves, not only for crossing.

Points close together *along* the path are exempt, being legitimately close in space, and the
exemption follows from the clearance rather than being a separate number: the tightest U-turn whose
two arms sit exactly the clearance apart has half of it as its radius, so its arc is a little over
`π` times the clearance.

Random paths are written to `Assets/Benchmarks/Paths/Random`. The driver's folder scan recurses, so a
batch joins every later sweep with no config edit — which cuts both ways: a batch left behind from an
earlier seed silently adds its runs to every sweep from then on, which is why the window clears the
folder by default and why `MoSynth > Benchmark > Delete Random Paths` exists.

Regenerating the same seed reproduces the geometry exactly, but not the file bytes: Unity assigns
fresh local fileIDs on every `SaveAsPrefabAsset`, so a regenerated batch always shows a diff even
when every knot is identical.

## Open paths

Open splines are no longer only a code path — the random families generate them. A run on one
finishes on reaching the far end (`SplineLapTracker.OpenSplineEndT`, 0.995) rather than by counting
laps, so `lapsRequired` is ignored and `completedLaps` reads as **the fraction of the path covered**,
not a lap count. The CSV column keeps its name, because renaming it would break every existing report.

Both spline inputs clamp rather than wrap there. The motion field's pure-pursuit input already did.
The constant-speed follower did not: it wrapped its parameter unconditionally, so about one prediction
horizon before the reference reached the end, the predicted trajectory teleported back to the start
and the character was handed features telling it to turn around. Its own projected parameter never
reached 0.995, so every open path would have been reported as a timeout with a spurious return leg
polluting the error metrics. `SplineControlInput.Fold` now wraps or clamps according to a virtual
`IsClosed` seam — virtual because `SplinePoseKeypointControlInput` follows a path that is not a
`SplineContainer` at all and has to answer for itself.

## Running a sweep

**One-time setup.** `MoSynth > Benchmark > Create Starter Assets` extracts path and character prefabs
from the example scene and writes a config; `MoSynth > Benchmark > Create Standard Paths` writes the
four paths above, and `MoSynth > Benchmark > Create Random Paths…` adds a seeded batch. Existing
per-prefab overrides are preserved on regeneration, and overwriting in place keeps asset GUIDs so a
config already pointing at a prefab survives.

**Interactive.** `MoSynth > Benchmark > Run Sweep`, or the inspector button. The config is resolved
from the selection, else the last one used, else the only one in the project.

**Headless.**

```powershell
.\Tools\run-benchmark.ps1
.\Tools\run-benchmark.ps1 -Config "Assets/Benchmarks/Ablation.asset" -Output "Benchmarks/ablation"
```

Unity must not already have the project open. The script never passes `-quit`, and that is
load-bearing: a sweep runs in play mode, so `SynthesisBenchmarkCli.Run` returns long before it
finishes and `-quit` would tear the Editor down mid-run. Batchmode Unity stays alive after
`-executeMethod` returns on its own, and the driver calls `EditorApplication.Exit` with the sweep's
status code once the report is written. Only headless runs exit; a developer running from a live
Editor wants their Editor back.

Note the script's defaults hardcode a project path and a specific Unity version, so it is not
portable as shipped — though it does check and fail cleanly.

## Output

```
<output>/
  results.csv        26 columns, flat, for spreadsheets
  results.json       the same rows plus machine context and the per-stage cost breakdown
  progress.jsonl     one row per run, appended as it finishes
  recordings/        <Method>_<Path>_<timestamp>.{bin,json}
  unity.log          headless runs only
```

Exit code 0 on a complete sweep, 1 on an abort or a cut-short run. `NaN` appears literally in the
CSV, which spreadsheet tools read as text rather than a blank.

Because everything is derived from the recordings, `BenchmarkRunEvaluator` can be re-run over stored
recordings without re-running the sweep.

## Source map

| Concern | File |
| --- | --- |
| Sweep state machine | `Assets/AnimationTools/Editor/Benchmark/SynthesisBenchmarkDriver.cs` |
| Entry points | `SynthesisBenchmarkLauncher.cs`, `SynthesisBenchmarkMenu.cs`, `SynthesisBenchmarkCli.cs` |
| Surviving the domain reload | `SynthesisBenchmarkPlan.cs` (via `Temp/`) |
| Metrics from a recording | `BenchmarkRunEvaluator.cs` |
| Report writing | `BenchmarkReportWriter.cs` |
| Path and starter-asset generation | `BenchmarkPathGenerator.cs`, `BenchmarkStarterAssets.cs` |
| Config and overrides | `Assets/AnimationTools/Runtime/Benchmark/` |
| Lap probe | `Assets/AnimationTools/Runtime/Benchmark/BenchmarkLapProbe.cs` |
| Headless wrapper | `Tools/run-benchmark.ps1` |

**Tests.** `MotionQualityMetricsCalculatorTests` and `SynthesisCostMetricsTests` cover the pure
calculators, including the nearest-rank percentile and the missing-contact-data case. The driver,
probe, evaluator and launchers are untested by design — they are Editor state machines and file I/O.
