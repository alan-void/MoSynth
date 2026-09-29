---
type: Architecture Guide
title: Motion field policies
description: The similarity metric, the candidate action set, and the two policies plus two debug modes the runtime steps the field with.
tags: [python, motion-field, knn, policy]
sources:
  - id: openwiki-source-13742752b942a8c72fc71381
    resource: repo://Assets/MotionField/MotionFieldStage.cs
  - id: openwiki-source-3d18ac229a9f725ecb715bab
    resource: repo://Python/motion_field/field.py
  - id: openwiki-source-cd81fde2d8ed86cd0b51960f
    resource: repo://Python/motion_field/trainer.py
generated: {by: "claude-code", at: "2026-09-29T08:27:12.326Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-29T08:27:12.326Z
---

# Motion field policies

`motion_field/field.py` is the runtime brain. It owns the similarity metric and its k-NN, the candidate
action set, one-step integration with drift correction, the two policies plus two debug playback
modes, the value-function *consumer* side, and the debug introspection the visualizer reads.

It does no file I/O of its own, and it does not fit anything — that is
[value function training](value-function-training.md).

## The similarity metric

Every database state is turned into a feature array of per-joint positions and velocities, weighted,
and compared by a **sum of per-joint L2 norms**.

Two invariances are built in, and both are deliberate:

**Frame invariance.** Forward kinematics pins the character frame to the origin with identity
rotation, so **where the character is in the world never influences which states it matches**.

**Root-translation invariance.** Both FK passes substitute the root bone's *rest-pose* offset for the
pose's own. The argument is specific: a live root offset is a rigid translation of *every* joint, so
it would enter all rows at once and **drown the per-joint shape signal this metric exists to
compare**. The exact offset does not matter — any constant cancels when two states are subtracted —
it only has to be the same for every state.

> **Changing this metric in code invalidates every trained field, by hand.** Unity's staleness check
> watches config *fields*, and an edit inside the metric function moves none of them. See
> [the staleness contract](../motion-field/config-and-training.md).

### Bone weights

Weights are **resolved by joint name**, because the feature rows follow a depth-first walk that
happens to match the serialisation order but is not guaranteed to. Resolving by name means a
reordered skeleton moves the weights with it instead of silently applying them to the wrong bones.

Rows for the character frame and the rig root are **forced to neutral**: the frame's position and
velocity are structurally zero, and the root bone's translation is replaced by a constant. Neither
can move a distance, so normalising them lets an otherwise-uniform table collapse to "no weights" and
skip the multiply entirely.

The k-NN is chunked because the difference tensor is quadratic in the chunk size — a few hundred MB
at the default chunk on this database.

## Actions

**One action per neighbour.** Take the similarity weights, pin that neighbour's weight to 1, and
renormalise. So each action leans on a different neighbour while still respecting the similarity
distribution.

Stepping an action blends the neighbours' velocities under those weights, integrates, and then tugs
the result toward an actual database state.

### The frame slot is zeroed every step

This is the subtlest thing in the file.

**The frame slot is a per-frame increment, not a world pose.** Database states are stored in their own
character frame, and the tug blends against one of them. So an accumulated world position would be
dragged toward the origin every single frame — a compounding collapse that reaches the origin in
**about 30 frames**.

Unity owns world placement anyway and integrates it from the returned velocities, so zeroing here
also makes training and runtime **structurally identical**: the frame of the returned pose is exactly
the increment this step produced.

### The tug compounds with the step size

The tug is a contraction applied **per call**, so its strength has to follow the step size.

At a `delta_time` shorter than one database frame, a fixed ratio would pull onto a database pose
faster than the scaled advance can move away — **a literal freeze**. Compounding it per database
frame makes *n* small steps tug exactly as much as one full frame, and is the identity when the step
equals the frame time, which is what training uses.

Its purpose is drift correction: pull a little toward a real database state so the field cannot wander
into regions it has no data for.

## The policies

| Method | Behaviour |
| --- | --- |
| `optimal_action` | scores each candidate with `R + γ·V(s′)`; falls back to greedy, silently, when no value function is loaded |
| `greedy_action` | picks whichever action scores best on the immediate reward alone: facing, travel direction and the locomotion bonus on the next frame |
| `get_next_pose` | debug: walk the database frame by frame from the nearest match |
| `get_next_pose_from_field` | debug: snap to the successor of the nearest state |

`optimal_action` **will accept a worse immediate heading when it sets up a better turn**. That
anticipation is the whole reason for training a value function.

The runtime scores with the **same Bellman objective the training optimised** — immediate reward plus
discounted value. Scoring on discounted future value alone would optimise something the value
function was never fitted to. For the same reason every term of the immediate reward — heading,
travel and the locomotion bonus, the last paid on the arrival state through the same neighbourhood —
is computed exactly as value iteration computed it, or the argmax drifts off-policy.

Because each action rotates the character, the goal moves in its frame, so the successor's value is
read at the shifted heading.

### Interpolation makes a discrete table continuous

The value function is stored only at discrete database states. `interpolate_value` reads it linearly
across the two bracketing headings on the task grid, then weights by motion-space similarity.

**That is what makes it behave as if it were continuous over the whole field.**

## The reward

```
theta' = wrap(theta + Δyaw)                                  # goal angle after the action
R      = -|theta'|                                           # face the goal
         - travel_factor · speed_gate · |wrap(beta + theta')|  # move toward it
         + locomotion_factor · moving_score(s')              # keep moving
```

### Why facing alone is not enough

The heading term only measures how far the character's **facing** is from the goal. On its own, any
action that faces the goal scores perfectly **whichever way it moves**, so a strafe or a backpedal
that faces the goal is as good as walking at it. That is the
[facing-versus-travel](../animation-tools/simulation-frame.md) distinction again, this time inside a
reward.

On a database with lateral material the policy uses that freedom. Measured on the Y Bot Holden
database, **25% of moving states travel more than 45° off their facing, and 6.5% more than 135°**.
With the heading term alone, the motion field strafed on the Oval and sharp-cornered benchmark paths:
it sped up to about 1.85 m/s against about 1.4 m/s elsewhere and slid 2–3 m off the path. Adding the
travel term at `travel_factor = 0.5` brought the mean path error over the 17-path Holden sweep from
**0.86 m to 0.10 m** and the mean heading error from **22.4° to 9.5°**.

### The travel term

`beta` is where the arrival state moves, measured from where it faces. It is read from the frame slot
of the packed arrival velocity, which is expressed in the character frame's own axes (+Z forward,
+X right), as `atan2(x, z)` — positive to the right, like Unity's `SignedAngle`.

**The sign.** `theta` is `-SignedAngle(facing, goal)` (see
[the stage](../motion-field/motion-field-stage.md)), so after the action the goal sits at `-theta'`
relative to the new facing. Travel sits at `beta` relative to that same facing. The gap between them
is `beta - (-theta') = beta + theta'`: zero walking straight at the goal, π/2 for a strafe that faces
it, π for backing toward it.

**The speed gate.** Below walking speed the direction of travel is noise, so the term is multiplied by
the same linear ramp the locomotion score uses. An idle state pays nothing; the locomotion bonus is
what keeps idling from being attractive.

**The heading term stays.** With the travel term alone the policy could walk at the goal with its
torso pointing elsewhere. The two terms together penalise both ways of going wrong.

Measuring `theta` against travel direction instead of facing was the rejected alternative. It needs
no extra term, but it goes unstable at low speed, exactly where the gate above switches the travel
term off.

`travel_penalty` holds the runtime's copy of the term. The trainer computes the same expression per
`(state, action)` from the post-tug arrival velocity, and the two **must agree**. See
[value function training](value-function-training.md).

### The locomotion score

The reference implementation scored states by locomotion-set membership — walk versus jog. With a
single set that generalises to **moving versus idle**, which is the distinction that actually matters:
it is what keeps an idle pose from being a zero-cost place for the reward's heading term to hide in.

It is **ramped rather than thresholded**, so slow turn-around frames keep partial credit instead of
flickering across a cliff.

The greedy policy needs the same bonus for a related reason: standing produces exactly zero yaw,
which beats the yaw wiggle of any real walk cycle once the character is aligned — so without the
bonus an idle-targeting action wins every tie.

## Yaw extraction

Yaw is read from the frame quaternion **exactly, not approximately**, because the frame rotation is
built as a pure yaw by construction and reduces to a two-component `atan2`.

> Note the **xyzw** component order. The wxyz formula the reference implementation uses would read the
> wrong components here and return nonsense.

## Loading a value function

`load_value_function` returns `False` — leaving the field in greedy mode — rather than raising when
the file is missing or unreadable, because Unity calls it inside `Py.GIL()` where an exception
surfaces as an opaque managed error.

**Whether the file matches the config is decided in Unity before the call, and nothing is re-checked
here.** The caller does have to get it right: every index in the value function is a row of the pose
database, so a file trained against a re-extracted `.mmpose` addresses the wrong poses **with no
symptom this module could detect**.

## Debug introspection

`get_debug_arrays` returns the neighbourhood, the weights, and the slot the policy chose. The chosen
slot indexes into the returned neighbour list, and it is **exactly the value handed to the step
function** rather than a re-derivation — so the Unity highlight cannot drift out of sync with the
policy.

Neither debug playback mode records a decision, so the arrays stay empty in those modes.

## Source map

`Python/motion_field/field.py` — the whole page.

**Things to be careful with:**

- Two docs reference `resolve_bone_weights`, a symbol that does not exist; the function is
  `pack_bone_weights`.
- `optimal_action` uses the neighbour count baked into the *trained file*, while `greedy_action` uses
  the config's. A config whose `kNeighbors` differs from the trained file's silently changes
  neighbourhood size when you toggle the policy.
- The two debug modes disagree about clip boundaries — one wraps modulo the state count, the other
  clamps — and neither is clip-aware, even though the state-index construction went to trouble to be.
- The feature's velocity half always uses the *database* frame time, never the render step. Correct,
  but easy to misread.
