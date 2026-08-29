---
type: Architecture Guide
title: Feature vectors and query masking
description: How an authored feature configuration becomes a fixed-width float vector per frame, and why an inactive query channel is weight-masked rather than filled.
tags: [features, query, normalisation, motion-matching]
sources:
  - id: openwiki-source-28684d61d3dab2bdc41d31f4
    resource: repo://Assets/MotionMatching/Editor/Core/MotionMatchingDataEditor.cs
  - id: openwiki-source-60a37e788745b5c0c6e5eb05
    resource: repo://Assets/MotionMatching/Runtime/AssemblyInfo.cs
  - id: openwiki-source-fed6ec6af0a6135c6cbeddce
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/MotionMatchingControlInput.cs
  - id: openwiki-source-f84c8bceda0edfaac6926af8
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingStage.cs
  - id: openwiki-source-504c316cf746c15242971108
    resource: repo://Assets/MotionMatching/Runtime/Features/FeatureSet.cs
  - id: openwiki-source-b6e9663124b6851f50898282
    resource: repo://Assets/MotionMatching/Runtime/Features/MmFeatureLayout.cs
  - id: openwiki-source-c6f33af525c79b0bf9fb187d
    resource: repo://Assets/MotionMatching/Runtime/Features/MmFeatureLayoutBuilder.cs
  - id: openwiki-source-aa542d59a533db81cc11c9fb
    resource: repo://Assets/MotionMatching/Runtime/Features/TrajectoryFeatureChannel.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Feature vectors and query masking

Motion matching compares a **query vector** describing what the character should do next against a
**feature vector** precomputed for every database frame. This page is about how those vectors are
built, laid out, normalised, and — most importantly — how parts of a query are switched off.

## Layout

The vector order is invariant: **all trajectory floats, then all pose floats.**

That falls out of the section-key mechanism in
[the layout system](../animation-tools/channel-layout-system.md). The two feature section keys sit
immediately after every built-in pose section, so trajectory comes first and the boundary between the
two blocks is simply the pose section's start offset.

| Feature type | Width |
| --- | --- |
| `TrajectoryFeatureChannel` | `floatsPerPrediction × predictionCount`, where a prediction is 1–3 floats after axis masking |
| `PoseFeatureChannel` | always 3 |

A trajectory feature samples a bone — or the character's derived simulation frame — a number of
frames into the future, stored once per entry of `predictionFrames`. A pose feature is a joint of the
current pose expressed in the character frame.

## Rebuild, do not mutate

A feature definition is **simultaneously the authored asset data and the layout channel**. The layout
keys its offset table on the descriptor, and a feature's hash covers its full authored content —
type, bone name, axis masks, every prediction frame.

So editing a feature that a built layout already holds changes its hash and makes its dictionary
entry unreachable. **Rebuild the layout after editing a feature**; never mutate one in place.

The constructor also rejects duplicates outright, so two identically-configured feature definitions
cannot coexist in one asset.

## Inactive channels are weight-masked, never filled

This is the most important behaviour on the page.

Bone trajectory channels are **optional constraints**. A control input switches one on by supplying a
target; the base implementation of `GetBoneTrajectoryFeature` returns `false`, meaning off.

```csharp
if (featureDef.simulationBone)
{
    controlInput.GetTrajectoryFeature(featureDef, p, character, feature);
}
else
{
    var active = controlInput.GetBoneTrajectoryFeature(featureDef, p, character, feature);
    for (var f = 0; f < featureSize; f++)
    {
        if (!active) feature[f] = 0f;
        _featureWeights[offset + f] = active ? _authoredFeatureWeights[offset + f] : 0f;
    }
}
```

Precisely:

- **Simulation-frame channels are always filled and never masked.** Only bone channels are optional.
- When a bone channel is off, its query floats are zeroed **and** its per-float weights are zeroed.
- When it is on, the weights are restored from `_authoredFeatureWeights`, a snapshot taken at `Init`.

The weight mask is what makes the channel not matter; zeroing the values is belt-and-braces. And
because `Init` handed the *same* `NativeArray` to the search backend, the mask reaches the Burst job
with no re-binding at all.

### Why not fill it with the current value?

The obvious alternative — when a foot constraint is inactive, fill it with where that foot *currently*
is — sounds neutral and is not.

It would make every candidate frame be scored on how closely its foot matches the foot the character
already has. That silently adds *"keep doing what you are doing"* to the search objective, **biasing
the search against switching**. A zero weight removes the dimension from the metric entirely; a
current-frame fill only pretends to.

The behavioural payoff is visible in
[spline pose keypoints](spline-pose-keypoints.md): away from a keypoint the channels stay off, the
stage masks them, and the search behaves like plain spline following. Near one they switch on with a
real bone target.

## Direction features bake facing

A `Direction` trajectory feature stores **the character's facing**, not its direction of travel.

- For a simulation-frame channel: the *future* pose's own simulation-frame forward.
- For a bone channel: that bone's rest-pose forward axis, rotated into character space.

In neither case is a position difference taken — travel direction never enters. Both are then
expressed in the *current* pose's character frame.

This is the database half of the facing/travel distinction. The query half — and what goes wrong if
you feed travel into it — is on [the simulation frame](../animation-tools/simulation-frame.md).

## Axis masking

Axes can be masked out, so one prediction is one to three floats wide. A masked axis is **not stored
at all**, so packing and unpacking step over it.

One asymmetry to watch: for a `Direction` feature the masked axes are zeroed *before* normalisation,
so a Y-masked direction is a unit vector **in the ground plane** — not the projection of a unit 3D
vector. `Position` features are neither pre-zeroed nor normalised; packing simply skips the masked
axis. So `zeroY` means two different things for the two feature types.

The editor **forces** a simulation-frame channel to mask Y, because the simulation frame is a
ground-plane frame. Such features are therefore always two floats (X and Z) per prediction. It also
constrains the three toggles so at least one axis survives.

## Normalisation

Extraction computes a mean and a standard deviation per valid frame, then normalises. Invalid frames
— those too close to the end of a clip for their lookahead — are left at zero and excluded from both
the statistics and the normalisation.

The mean is per dimension. **The standard deviation is not.** For each feature, the per-dimension
deviations are averaged across all of its floats and that single value is written back to every one
of them.

That is deliberate: it preserves the feature's internal shape rather than independently rescaling
each axis. It is also easy to misread, because the array is sized per float and the normalisation
loop is written per index.

At query time only the trajectory block is normalised, because the pose block is copied straight out
of the already-normalised database.

## Bone resolution

A simulation-frame feature is given bone index **−1** rather than a real bone, deliberately: a
simulation-frame feature is derived from the whole pose, so anything that indexes a bone with it is
wrong and should fail loudly.

A non-simulation feature whose bone cannot be resolved gets the opposite treatment — a warning and a
fallback to the skeleton root.

## Baking

`GenerateDatabases` on the `MotionMatchingData` inspector runs a fixed order, and the order is
load-bearing because each step depends on the last:

1. extract poses,
2. serialize them to `.mmpose` — which is also what [the Python side reads](../python/pose-data-bridge.md),
3. compute joint forward axes,
4. extract the features derived from those poses,
5. serialize those to `.mmfeatures`.

The asset is a **recipe, not the data**. Edits take effect only when the databases are regenerated,
and stale generated files are not detected. See [on-disk formats](../animation-tools/on-disk-formats.md)
for why `.mmfeatures` is a separate file and what "unversioned" costs.

## Assembly boundary

`Runtime/AssemblyInfo.cs` declares `[assembly: InternalsVisibleTo("MotionMatching.Editor")]`. Worth
knowing it is there — but nothing currently exercises it: the editor reaches everything it needs
through public API and serialized-property name lookups, and the one `internal` in this area is a
constructor used within the same assembly.

## Source map

| Concern | File |
| --- | --- |
| Storage, validity, normalisation, BVH bounds | `Assets/MotionMatching/Runtime/Features/FeatureSet.cs` |
| Offsets and handles | `Assets/MotionMatching/Runtime/Features/MmFeatureLayout.cs` |
| Section keys, bone resolution | `Assets/MotionMatching/Runtime/Features/MmFeatureLayoutBuilder.cs` |
| Trajectory features | `Assets/MotionMatching/Runtime/Features/TrajectoryFeatureChannel.cs` |
| Pose features | `Assets/MotionMatching/Runtime/Features/PoseFeatureChannel.cs` |
| The matching-specific interface | `Assets/MotionMatching/Runtime/Features/IMatchingFeature.cs` |
| `.mmfeatures` | `Assets/MotionMatching/Runtime/Features/FeatureSerializer.cs` |
| Authoring and baking | `Assets/MotionMatching/Editor/Core/MotionMatchingDataEditor.cs` |

**Tests.** `MmTestData` is the only file in the `MotionMatching.Tests` assembly and contains no tests
— it is a fixture helper (a four-bone synthetic skeleton, a deterministic random pose filler, and an
opt-in loader for the checked-in demo database) consumed by suites elsewhere. **No invariant on this
page is asserted by a test.**

One stale reference to ignore: both feature channels' rebuild-don't-mutate remarks name `PoseLayout`
in a `<see cref>`. The rule is correct but the type is wrong — these channels are held by
`MmFeatureLayout`, which shares the same base class.
