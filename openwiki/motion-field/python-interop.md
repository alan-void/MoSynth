---
type: Architecture Guide
title: Reaching Python
description: Two mutually exclusive transports to the motion field — an embedded CPython interpreter and an out-of-process ZeroMQ socket — and the constraints each imposes.
tags: [pythonnet, interop, threading, zeromq]
sources:
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-9b84862940b622d8527df945
    resource: repo://Assets/MotionField/MfConnector.cs
  - id: openwiki-source-839acd5f9c92d76d722dddc3
    resource: repo://Assets/MotionField/PythonRuntime.cs
  - id: openwiki-source-0d9ae15ae536e3580049e519
    resource: repo://Python/test_server.py
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Reaching Python

The motion field is evaluated in Python. There are two ways to get there, and they are alternatives
rather than layers.

| | `PythonRuntime` + `MotionFieldStage` | `MfConnector` |
| --- | --- | --- |
| Transport | embedded CPython via PythonNET | ZeroMQ request/reply, JSON on the wire |
| Cost | interpreter lives in the Unity process | latency plus a serialization round trip |
| Benefit | no serialization, direct array handoff | Python side restartable and debuggable without taking Unity down |
| Status | the live path | exploratory; see the drift below |

## One interpreter, one bootstrap

`PythonRuntime` is the single CPython bootstrap for everything on the MotionField side.

**The interpreter is process-wide and can only be configured once.** So the runtime stage and the
editor trainer must both come through here rather than each calling `PythonEngine.Initialize()` with
their own paths — pythonnet throws if the DLL path changes after the interpreter starts. A latch
enforces first-call-wins.

`EnsureInitialized` is safe to call repeatedly and does three things:

1. Assigns the CPython DLL path, if this is the first call. Both the DLL path and the virtual
   environment are serialized on `MotionFieldConfig`; an empty DLL path falls back to PythonNET's own
   `PYTHONNET_PYDLL` environment variable.
2. Adds the venv's `site-packages`, which is what supplies numpy, scipy and torch.
3. Appends the repository's `Python/` folder to `sys.path`, idempotently.

Both failure paths are typed and name the fix location — *"Set it on the MotionFieldConfig."*

**Python 3.13 is required** for PythonNET compatibility.

There is no shutdown. `PythonEngine.Shutdown()` is never called anywhere, so the interpreter lives for
the process — consistent with the configure-once argument, but it means the Editor and play mode
share one interpreter state.

## Module reloading drops the whole graph

`InvalidateProjectModules` removes every loaded module whose file lives under the project's `Python/`
folder. It replaces a per-module `importlib.reload`, and the reason is concrete:

> `importlib.reload` only re-executes the one module it is handed. **Its dependencies stay cached**,
> so reloading `MotionField` after adding a symbol to `motion_field_io` re-runs the new import line
> against the old dependency and dies on `ImportError` — editing a shared module was effectively
> impossible without restarting the editor.

There is a second reason, subtler and just as important:

> Dropping the graph wholesale means a group of imports issued after one call sees **a single
> consistent generation of the code**. Reloading module by module does not: each reload rebinds only
> that module's classes, so two modules can end up holding different class objects for the same
> class.

Hence the usage rule: call it **once before a group of related imports**, not on each one. That is
exactly what the stage does at `Init`, and it is why the embedding module is imported *without*
reloading afterwards — re-invalidating there would hand it a second copy of the field's classes.

Live Python objects created before the call keep working; they hold a reference to a class that is
simply no longer the one a fresh import returns. So only call it where everything downstream is about
to be rebuilt.

The invalidation script is kept as embedded source rather than a `.py` file in the scripts folder, for
a reason that is almost a joke and entirely correct: **a module that purges the project's modules
should not be one of the modules it purges.**

## The out-of-process alternative

`MfConnector` is a `MoSynthStage` that drives the pose from a motion field running in a **separate
Python process**, reached over a ZeroMQ request/reply socket with JSON poses on the wire.

The tradeoff is stated plainly: the socket costs latency and a serialization round trip, and buys a
Python side that can be restarted and debugged without taking Unity down — useful while that side is
still being written.

### The threading contract

This is the part to get right if you touch it.

**NetMQ sockets are not thread-safe and must not be closed from Unity's main thread.** So the socket
is created, used and disposed **entirely on the worker thread**. The two threads meet only at:

- two `ConcurrentQueue`s — one for outgoing delta-time requests, one for incoming poses,
- a `volatile bool` asking the worker to stop (also cleared by the worker when it dies),
- a `volatile string` carrying the worker's error, reported and cleared on the main thread,
- an interlocked 0/1 latch making `Dispose` idempotent.

No lock, no synchronisation context, and no Unity API touched from the worker.

**At most one request is outstanding.** Partly because a request/reply socket must receive its reply
before sending again, and partly because queueing one per frame would grow the queue every frame
whenever the server is unavailable.

The receive loop **polls with a short timeout** rather than blocking, so play-mode shutdown can be
observed between attempts — which is what bounds the shutdown wait and makes the two-second join
safe.

When no reply has arrived, `Apply` returns `false` and leaves the character in its previous pose
rather than a half-updated one. Note that under the
[stage contract](../animation-tools/synthesis-pipeline.md) this also terminates the pipeline for that
tick, silently skipping every downstream stage whenever a reply is late.

### It has drifted from its server

Be aware before using this path:

- **`MfConnector` hardcodes its desired direction** to a constant forward vector and never reads a
  control input, so it can only ever walk forward in the server's frame.
- It carries **no `MotionFieldConfig`**, so it cannot participate in the staleness contract at all.
- It sends a JSON object with a direction and a delta time; the only Python server in the repository,
  `test_server.py`, expects a **bare frame number** and would reply with an error object every tick.
- That server hardcodes 23 joints; the shipped database has 85.
- It logs on both the request and the reply path, unconditionally — one to two console lines per
  frame.

Treat `test_server.py` as the written record of the **reply schema** and nothing more. See
[the Unity call surface](../python/unity-call-surface.md).

## Source map

| Concern | File |
| --- | --- |
| Interpreter bootstrap, module invalidation | `Assets/MotionField/PythonRuntime.cs` |
| Out-of-process transport | `Assets/MotionField/MfConnector.cs` |
| Python-side counterpart | `Python/action_predictor.py`, `Python/test_server.py` |

**Tests.** None.
