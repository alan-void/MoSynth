---
type: Architecture Guide
title: Value function training
description: Fitted value iteration over the whole database, what the produced artefact does and does not record, and why the loader deliberately checks nothing.
tags: [python, training, value-iteration, format]
sources:
  - id: openwiki-source-93fd80dba91ea0f5bb6027e9
    resource: repo://Python/motion_field/io.py
  - id: openwiki-source-cd81fde2d8ed86cd0b51960f
    resource: repo://Python/motion_field/trainer.py
generated: {by: "claude-code", at: "2026-09-21T19:17:12.006Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-21T19:17:12.006Z
---

# Value function training

`motion_field/trainer.py` fits `V(state, goal heading)` over the whole database by **fitted value
iteration**, following Lee et al. 2010, section 6. `motion_field/io.py` owns the `.mffield.npz`
format and a deliberately thin load path.

Neither owns staleness policy — that is
[Unity's job](../motion-field/config-and-training.md).

## Why fit anything

The controller needs to know more than which action looks best right now. **Turning a walk cycle
takes several steps of setup, so a one-step-greedy policy oscillates instead of committing.**

Training fits a value function over the whole database, after which the runtime can act on
`Q = R + γ·V(s′)` and anticipate.

## The pipeline

1. **Precompute**, for every database state and each of its *K* candidate actions, the yaw that
   action produces and the neighbourhood of the state it lands in. This is the expensive part, and it
   is **rebuilt on every train**.
2. **Iterate** the Bellman backup over that fixed transition structure until the residual stops
   moving.
3. **Write** the value function to StreamingAssets.

Both the task parameter — the goal heading, quantised onto a grid — and the motion state are
interpolated, so although values are stored only at discrete database states **the result behaves as a
continuous function over the field**.

The transition tables are intermediate and never persisted. Only the value function ships.

### The backup

```
bonus[s,a]    = factor * Σ_j w[s,a,j] * state_scores[s'[s,a,j]]
reward[s,a,t] = -|wrap(theta[t] + delta_yaw[s,a])| + bonus[s,a]
Q[s,a,t]      = reward + gamma * Σ_j w[s,a,j] * V(s'[s,a,j], theta[t] + delta_yaw[s,a])
V[s,t]        = max over a of Q[s,a,t]
```

Two things in there are load-bearing:

**The bonus is not optional.** The heading term is how badly the character faces away from the goal
after the action, and on its own it is **never positive** — which makes any idle state the reward can
reach a perfect score once aligned. The bonus, keyed on root speed and paid on the *arrival* state, is
what makes continuing to move strictly better than freezing.

**The successor's value is read at a shifted heading.** Because the action rotates the character, the
goal moves in its frame — so the lookup is at `theta + delta_yaw`, not at `theta`, interpolated
across the two bracketing grid headings.

The task grid is half-open: `theta_count` headings spanning `[-π, π)`, built by sampling one extra
point and dropping the wrap-around duplicate. It must match what the runtime policy uses, which is
why it is one of the three scalars persisted.

`gamma` also bounds V at `1 / (1 - gamma)` per unit of per-step reward.

## Performance choices

**Fully batched.** Integrating one action at a time the way the reference implementation does costs
roughly **5 minutes** here; the batched form takes about **15 s**. The `(state, action)` pair is
flattened into one axis so a chunk is a single batched integrate rather than many separate ones.

**Chunked by state.** The dominant allocation is the neighbour gather — about **46 MB** at 512 states
with K = 15. Gathering all **7,838** states at once would need roughly **700 MB**.

Each action is tugged toward the neighbour it emphasises, **which is what keeps the K outcomes
distinguishable from one another**.

## The artefact

`.mffield.npz` holds the value table and a residual log, plus **exactly three scalars the runtime
cannot recover on its own**: the heading grid the values are sampled on, and the `gamma` and
`k_neighbors` the fit assumed — which the policy has to reuse for its one-step lookahead to score
actions the way training scored them.

It records **nothing about the run that produced it**, and that is deliberate:

> Deciding whether a value function still matches the config that will run it is Unity's job, not this
> module's. `hasTrained` is set by the Train button and cleared the moment a field is edited, so the
> answer is known **in the inspector** rather than at load time with the character already in play
> mode. The loader therefore takes whatever file it is handed, and the file carries only what the
> loader reads back — since nothing would consult anything else.

`load_value_function` returns `None` rather than raising on a missing or unreadable file, because
callers run inside `Py.GIL()` from Unity where an exception surfaces as an opaque managed error, and
an unusable value function **should degrade to greedy control rather than take the player down**.

There is no format version, so a file written by an older layout shows up as a missing-key error —
the first key the loader reaches for that is not there.

## What is *not* recorded, and why it matters

The three persisted scalars are not the full set of things training assumed. `locomotionFactor`,
`locomotionSpeedThreshold`, `posWeight`, `velWeight`, `tugRatio` and the bone weights all shape the
fit and **none of them is written to the file**.

The runtime re-applies the locomotion bonus and re-computes the metric from the config's *current*
values. Change one without retraining and the runtime's Q silently disagrees with the fitted V.

`hasTrained` is the only thing standing between you and that, which is exactly why it
[over-flags](../motion-field/config-and-training.md).

## Convergence

The residual log is preallocated per epoch and sliced to the number actually run. The final entry is
the mean residual of the last epoch — so a run that hit the tolerance reports a value *below* it,
while a run that exhausted its epoch budget reports whatever it reached, **with no flag
distinguishing convergence from exhaustion**. The returned summary's epoch count is the only way to
tell.

The residual tolerance is not exposed by the Editor button, so in practice it is always the default.

## Source map

| Concern | File |
| --- | --- |
| Precompute, value iteration, the entry point | `Python/motion_field/trainer.py` |
| The `.npz` format and its loader | `Python/motion_field/io.py` |
| The Editor button that drives it | `Assets/MotionField/Editor/MotionFieldConfigEditor.cs` |

> **The advertised command line does not work.** The module docstring shows a standalone invocation,
> but `main()` and its `__main__` guard are both **commented out** — running the file does nothing,
> and the example data directory in that docstring does not exist. Train from the Editor button.

The embedding is a separate artefact produced by a separate module and a separate button, sharing
only the `MotionField` object and its metric. It is independent of training: the value function does
not need it, and deleting it only turns the visualizer off. See
[the pose manifold](../motion-field/pose-manifold-embedding.md).

**A naming trap.** "Scores" means two unrelated things in this area: `ValueFunctionData.scores` is the
per-epoch residual log, while `MotionField.state_scores` is the per-state locomotion score. The
trainer shadows one with the other locally.
