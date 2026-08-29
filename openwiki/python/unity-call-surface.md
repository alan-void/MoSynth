---
type: Architecture Guide
title: The Unity call surface
description: What crosses the C#/Python boundary in each direction every tick, the virtual root the packed layout needs, and the debugger attach that runs by default.
tags: [python, pythonnet, marshalling, interop]
sources:
  - id: openwiki-source-9b84862940b622d8527df945
    resource: repo://Assets/MotionField/MfConnector.cs
  - id: openwiki-source-18938fea788ba704e8dbb6d7
    resource: repo://Python/action_predictor.py
  - id: openwiki-source-879d0c2123931699bca99dc8
    resource: repo://Python/debugging/python_net.py
  - id: openwiki-source-0d9ae15ae536e3580049e519
    resource: repo://Python/test_server.py
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# The Unity call surface

`action_predictor` is the module `MotionFieldStage` holds and calls. It owns two directions of
traffic:

- **`load_animations`** — read the database Unity exported and repack it into the `(x, v, y)` triples
  the motion field is defined over.
- **`get_pose_arrays`** — flatten a synthesized pose into lists PythonNET can marshal back into C#.

This is the Python-side counterpart of [reaching Python](../motion-field/python-interop.md).

## What crosses, and when

**At init**, once:

```
load_animations(dataPath, name)
  → (skeleton, pose_x, pose_v, pose_y, pose_contacts, frame_time, pose_set)
MotionField(pose_x, pose_v, pose_y, skeleton, frame_time, device=…, pos_weight=…,
            vel_weight=…, k_neighbors=…, tug_ratio=…, knn_chunk=…, bone_weights=…,
            locomotion_factor=…, locomotion_speed_threshold=…)
load_value_function(path, log_callable)          # optional
load_embedding_arrays(path, stateCount, log)     # optional, debug only
```

**Per tick**, exactly one policy call plus one pose handover:

```
optimal_action | greedy_action | get_next_pose | get_next_pose_from_field
  (theta, current_x, current_v, delta_time)  → (new_x, new_v)
get_pose_arrays(skeleton, x, v, contacts)
  → (positions, quaternions, linear_velocities, angular_velocities, leftContact, rightContact)
get_debug_arrays()                                # when collecting debug data
```

`new_x` and `new_v` stay on the Python side as opaque handles between frames — they never round-trip
through C#.

Note that **the policy itself never crosses**. C# picks the method name at the call site; no enum,
string or integer is sent.

## The virtual root

The packed layout reserves slot 0 for the character frame, and **no rig bone corresponds to it** —
it is where the per-step translation and yaw the state is invariant to live.

So `with_virtual_root` copies the skeleton under an extra parentless joint. Giving the joint tree a
matching joint keeps tree and array aligned, so **bone weights, feature rows and FK indices all refer
to the same bone as the slot beside them**, and FK pins slot 0 to the origin as that invariance
requires.

It is **copied rather than reparented in place**, so the pose set's own skeleton goes on describing
the pose set's own frame-free arrays. The returned skeleton is therefore one bone longer than the
file's.

## Why the last frame of every clip is dropped

A state at frame *i* needs both its own velocity **and** the next frame's velocity, which becomes
`pose_y` — the thing the field learns to predict. Frame *i+1* must therefore belong to the same clip.

So the last frame of every clip is dropped. Otherwise **a state would predict its successor from the
first frame of an unrelated animation.**

## What invariance buys

Each usable frame becomes a triple: the pose (`pose_x`), how it is moving now (`pose_v`), and how it
is moving one frame later (`pose_y`).

`pose_x` is made **translation- and yaw-invariant**, so "walking north at the origin" and "walking
east across the map" are the same state. That motion is not lost — it lives in `pose_v` as a rate in
the character frame.

Rates are stored **per second**, not per frame, so integrating them requires an explicit time step.
Angular rates stay as rotation vectors, for [the aliasing reason](pose-data-bridge.md).

`pose_y` is why the frame reconstruction has to extend one frame past the end of a clip: frame *i+2*
can be one past the end, and only *i+1* is guaranteed to be inside it.

## The marshalling contract

`get_pose_arrays` returns **plain Python lists rather than numpy arrays**, because PythonNET marshals
those directly.

The arrays are **indexed by rig bone**, so they are one shorter than the packed layout's joint count.
C# stores no character frame — it derives one from bone 0 exactly the way the Python side does.

The pose handed over is already expressed in its own frame, so its derived frame is the identity.
Which means bone 0 simply carries the frame-local root state, and **the frame's own rates are folded
into bone 0's velocities** — which is where the C# side reads them back out.

The cross term in that fold is the root swinging about the frame origin, and it vanishes whenever the
frame is derived from the root itself, since the root then sits directly above the origin.

Every non-root bone gets zero linear velocity, which is consistent with C# reconstructing bone
positions from rest offsets — but it does mean the velocity array crossing the boundary is meaningful
for bone 0 only.

## The debugger attach

Importing `action_predictor` attaches to a PyCharm debug server, via `debugging/python_net.py`.

The cost argument for gating it is real: attaching costs a socket timeout per import and injects a
debugger egg into `sys.path`, which is **pure overhead for batch work such as training**.

> **The comment says "Opt in." The implementation is opt-out.** The environment variable defaults to
> enabled and the test is *not equal to zero*, so the attach runs on every import — including training
> and embedding runs — unless it is explicitly disabled. The retry backoff bounds, but does not
> remove, the per-import cost when no debug server is listening.

The attach logic itself is careful for a specific reason: **PyCharm clears pydevd's global debugger
when its debug-server socket closes**, so a healthy connection is left alone, including across Unity
domain reloads. When it must reconnect it drops the whole pydevd module graph, because pydevd stores
the socket in module globals and a new server needs a fresh graph — restoring the stdout/stderr
redirectors first so the new session owns them. Failures back off on a timer rather than flooding the
console, since a debug server is normally absent outside a debugging session.

Both the egg path and the debug port are hardcoded.

## test_server.py

This is the standalone ZeroMQ server, and its real value is as **the only written record of the wire
schema** for [`MfConnector`](../motion-field/python-interop.md):

| Key | Shape |
| --- | --- |
| `JointLocalPositions` | array of `{x, y, z}` |
| `JointLocalRotations` | array of `{value: {x, y, z, w}}` |
| `JointLocalVelocities` | array of `{x, y, z}` |
| `JointLocalAngularVelocities` | array of `{x, y, z}` |
| `LeftFootContact`, `RightFootContact` | bool |

Port 5555, matching the client's default. The nested `value` on rotations mirrors Unity.Mathematics'
`quaternion`, whose only field is a `float4 value`.

**The request side has drifted.** This server expects a bare decimal frame number; the C# client
sends a JSON object carrying a direction and a delta time. Against the current client it would take
its error path and reply `{"error": "Invalid frame number"}` every tick — which the client would then
deserialize into a pose with all-null arrays.

It also hardcodes 23 joints where the shipped database has 85. Treat it as documentation of the reply
format and nothing more.

## Source map

| Concern | File |
| --- | --- |
| The per-tick surface, repacking, virtual root | `Python/action_predictor.py` |
| Debugger attach | `Python/debugging/python_net.py` |
| ZeroMQ reply schema | `Python/test_server.py` |

**One default to be careful with.** `load_animations`' hardcoded default data directory points at the
**motion matching** database, not the motion field one. Unity never uses it — the stage and the
trainer always pass the config's own asset path — but a notebook or standalone invocation that
relies on the default will silently train against a different database.
