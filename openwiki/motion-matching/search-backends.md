---
type: Architecture Guide
title: Search backends
description: The strategy seam for finding the closest database frame, the bounding-volume hierarchy that accelerates it, and the tag mechanism that is designed but unwired.
tags: [search, burst, bvh, performance, extension-points]
sources:
  - id: openwiki-source-c58a2c45f72f325c9cda92ec
    resource: repo://Assets/MotionMatching/Runtime/Core/Burst/BVHMotionMatchingSearch.cs
  - id: openwiki-source-b8191d782d2dcd28217e888b
    resource: repo://Assets/MotionMatching/Runtime/Core/Burst/LinearMotionMatchingSearchBurst.cs
  - id: openwiki-source-9608fa457c4aae6b1f8f1abe
    resource: repo://Assets/MotionMatching/Runtime/Core/Burst/TagOperationsBurst.cs
  - id: openwiki-source-75cfe8f91d76df4d805a4531
    resource: repo://Assets/MotionMatching/Runtime/Core/Burst/UtilitiesBurst.cs
  - id: openwiki-source-750b5bdfd0a423c5ab2468a3
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingSearch/BvhMotionMatchingSearch.cs
  - id: openwiki-source-f94e76bb9933b7c4eb42116e
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingSearch/LinearMotionMatchingSearch.cs
  - id: openwiki-source-c7050471458b93768d777c1b
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingSearch/MotionMatchingSearch.cs
  - id: openwiki-source-f84c8bceda0edfaac6926af8
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingStage.cs
generated: {by: "claude-code", at: "2026-08-29T23:30:04.693Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-29T23:30:04.693Z
---

# Search backends

`MotionMatchingSearch` answers one question: given a query feature vector, which database frame is
closest? Subclass it to add a new method — brute force, an acceleration structure, a learned index —
without touching [the stage](matching-stage.md).

It is one of the four `[SerializeReference]` seams, so a concrete search must be `[Serializable]`
with a parameterless constructor. Two ship, and `BvhMotionMatchingSearch` is the default.

## The contract

```csharp
public virtual  void Initialize(FeatureSet featureSet, NativeArray<bool> tagMask,
                                NativeArray<float> featuresWeights);   // base throws
public abstract int  FindBestFrame(NativeArray<float> queryFeature, float currentDistance);
public virtual  void Dispose();
```

`Initialize` runs once from the stage's `Init`, `FindBestFrame` once per search tick, `Dispose` at
the end. The three arrays stay alive for the stage's lifetime, so implementations **store rather than
copy them** — which is precisely what lets the stage's per-search weight mask reach the Burst job
with no re-binding.

`currentDistance` is the distance to the frame already playing, or `MaxValue` when it is unusable. It
seeds the running best, so candidates worse than it are rejected and a query that nothing beats
returns **−1**, meaning "keep playing what is playing".

Both implementations schedule their job and immediately `Complete()` it. The search is synchronous
within the tick; nothing overlaps.

## One metric, by convention rather than by code

Every implementation must use the same distance metric, because the caller computes
`currentDistance` itself and compares the result against it. If a backend used a different metric the
two numbers would not be comparable and "better than what's playing" would be meaningless.

The metric is **weighted squared distance** — squared rather than true distance because only the
ordering matters, and because it lets a search abandon a candidate early.

Worth being precise about how that contract is actually enforced: **it is not shared code.**
`MotionMatchingStage.SqrDistance` has exactly one call site — the stage scoring the current frame —
and neither Burst job calls it, because a job cannot take a `ReadOnlySpan<float>`. Both jobs
re-implement the same weighted sum of squares inline, with opposite operand order and identical
results after squaring.

So this is a discipline to uphold when adding a backend, not something the compiler checks.

## Linear: the reference

`LinearMotionMatchingSearch` scores every valid frame. Slower, but with **no acceleration structure
to get wrong**, which makes it the reference to check the BVH backend against when a search starts
returning surprising frames.

It goes further than merely being simple: its inner dimension loop deliberately has **no early exit**
once the running distance passes the best so far, so every candidate is scored in full. The reference
implementation forgoes even the safe optimisation, so that it cannot diverge for a subtle reason.

## BVH: index-based, not spatial

`BvhMotionMatchingSearch` skips whole runs of the database using a two-level hierarchy of
axis-aligned bounding boxes over the feature vectors.

The key design choice: **each box covers a run of consecutive frames rather than a spatial cluster.**
That is cheap to build and cheap to index — box membership is pure integer division — and effective
because consecutive frames have similar features anyway.

```
box k of a level spans frames [k * size, (k + 1) * size)
LargeBVHSize = 64
SmallBVHSize = 16
```

Traversal is three nested loops, one per level: large box, small box, frames. Each level applies the
same test — **the distance to a box's nearest point is a lower bound on the distance to anything
inside it**, so once that bound reaches the best distance so far, the whole box is skipped. The
dimension loops break early for the same reason: the running sum only grows.

The lower bound per dimension is `query − clamp(query, boxMin, boxMax)`, weighted and summed, which
is zero when the query lies inside the box's extent on that axis.

### Two things to keep true

**The nesting invariant is unenforced.** Small boxes nest inside large ones only because 64 is a
multiple of 16. The search relies on that. Neither value is a `const` — they are
`static readonly int` — so it cannot be checked at compile time. Keep it true.

**The boxes include invalid frames.** The bounds builder loops over every frame with no validity
check, so boxes containing invalid (all-zero, un-normalised) frames are inflated toward the origin.
Pruning stays *correct* — the bound is still a lower bound — but it is weaker than it could be.

The boxes belong to the `FeatureSet` and are built lazily on first request, so the search does not
dispose them. They are cached on a `IsCreated` check alone, so re-extracting features onto a live
`FeatureSet` would leave them stale.

## Tags: designed, not wired

`TagMask` is one flag per frame; false excludes it. It is how tag queries — "only walking frames" —
would narrow the database.

**Nothing populates it.** `SetTagBurst` and `DisableTagBurst` are never scheduled from anywhere, and
the stage allocates the mask all-true at `Init` and never rewrites it. The mechanism is complete and
inert.

The one piece of reasoning worth preserving from it: when a tag query *is* applied, each range is
**trimmed at both ends** — by the maximum prediction lookahead at its end and the maximum history at
its start — because a feature vector encodes the frames around it, so the edge frames of a range have
no valid trajectory and would let the search pick a pose whose trajectory runs off the tagged region.

This connects to a matching dead end upstream: nothing writes tags during extraction either. See
[the pose database](../animation-tools/pose-database.md).

Two smaller acknowledged gaps live in the job source: the per-frame `Valid` array would be
unnecessary if all features were valid, the tag mask could be a bitmask, and the tag filter currently
prunes nothing at box level — only per frame.

## Not a search backend

`UtilitiesBurst.cs` lives in the same `Runtime/Core/Burst/` folder and has nothing to do with
searching. It is iterative point-to-ellipse geometry following Eberly's *Distance from a Point to an
Ellipse*, used for crowd avoidance: a moving character's personal space is an ellipse — longer along
its direction of travel than across it — so "how close am I to stepping into someone" is a
point-to-ellipse distance query.

Nothing in the search path references it. It belongs conceptually with
[control inputs](control-inputs.md).

Its own arguments are worth keeping where it is documented: an ellipse has no closed-form nearest
point, so the exact method is bisection with a bracket chosen so a root is guaranteed inside it;
the solver handles only the first quadrant because an ellipse is symmetric about both axes, so
callers fold the query in and unfold the result; and the fast variant samples the circumference
instead of solving, which is good enough for steering because steering only needs to *rank*
obstacles.

## Source map

| Concern | File |
| --- | --- |
| The seam | `Assets/MotionMatching/Runtime/Core/MotionMatchingSearch/MotionMatchingSearch.cs` |
| Brute force | `.../LinearMotionMatchingSearch.cs`, `Runtime/Core/Burst/LinearMotionMatchingSearchBurst.cs` |
| BVH | `.../BvhMotionMatchingSearch.cs`, `Runtime/Core/Burst/BVHMotionMatchingSearch.cs` |
| Tag mask jobs | `Runtime/Core/Burst/TagOperationsBurst.cs` |
| Ellipse geometry (unrelated) | `Runtime/Core/Burst/UtilitiesBurst.cs` |

For the seam mechanism itself, see
[the synthesis pipeline](../animation-tools/synthesis-pipeline.md).

**Tests.** None. No test in the repository touches any search type or Burst job.

**Disposal.** `Dispose` is never called on either backend, because the stage overrides no
`OnDestroy`. The search result array uses `Allocator.Domain`, so it is reclaimed on domain unload
rather than leaked indefinitely — but the contract is not being honoured. That array is also length 2
with only element 0 ever written.
