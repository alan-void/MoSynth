---
type: "Reference"
title: "Auto-tagging: run folder contract and hazards"
description: "The JSON and stdout contract between the Unity and Python halves of auto-tagging, and the edits that silently break it."
sources:
  - id: openwiki-source-11f9c5768a67637f682ea8cc
    resource: repo://Assets/AnimationTools/Editor/AutoTag/AutoTagAnnotateProcess.cs
  - id: openwiki-source-a32b8a8cb70c029e8d2d1408
    resource: repo://Assets/AnimationTools/Editor/AutoTag/AutoTagApply.cs
  - id: openwiki-source-6af061b91822651dcb24aa62
    resource: repo://Assets/AnimationTools/Editor/AutoTag/BurnInLabel.cs
  - id: openwiki-source-d2f445007fb2bf05eaa40fcf
    resource: repo://Assets/AnimationTools/Editor/AutoTag/CapsuleFigure.cs
  - id: openwiki-source-5e72ad60c7de05fc009e0c48
    resource: repo://Assets/AnimationTools/Editor/AutoTag/ClipVideoRenderer.cs
  - id: openwiki-source-88c3b2718587d20ac4831d81
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/BoneNameConventions.cs
  - id: openwiki-source-4c126daabb8ebe9a24e533eb
    resource: repo://Python/autotag/annotate.py
  - id: openwiki-source-2b64687facac60f72d29bd87
    resource: repo://Python/autotag/gemini.py
  - id: openwiki-source-d0d4f38644e7bc857c687bb0
    resource: repo://Python/tests/test_autotag.py
generated: {by: "claude-code", at: "2026-09-26T20:15:29.621Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-26T20:16:03.254Z
---

# Auto-tagging: run folder contract and hazards

Human overview: [`animation-tools/auto-tagging.md`](../../animation-tools/auto-tagging.md).

Code: `Assets/AnimationTools/Editor/AutoTag/` (Unity, namespace `AnimationTools.Editor`) and
`Python/autotag/` (the CLI). The two halves share nothing but the contract below, so a change to
either side must be made on both, and both test suites cover their half of it
(`AutoTagApplyTests`, `Python/tests/test_autotag.py`).

## Run folder

`Library/AutoTag/<yyyyMMdd-HHmmss>/`:

| File | Writer | Reader |
|---|---|---|
| `manifest.json` | `AutoTagManifest` (render) | `autotag.manifest.load_manifest` |
| `videos/<clipGuid>.mp4` | `ClipVideoRenderer` | `autotag.gemini` (upload) |
| `raw/<clipGuid>.json` | `autotag.annotate` — `{model, responses: [{text, inputTokens, outputTokens}]}`, one entry per request, written *before* parsing | `--resume` |
| `results.json` | `autotag.annotate` | `AutoTagApply` |

JSON keys are camelCase on both sides. Every frame number in both files is **clip-local** (the
numbering tag toggles use), never slice-local. Segments in `results.json` are half-open, sorted,
non-overlapping, non-adjacent and inside the slice; Python's `postprocess.clean_segments`
guarantees that, and `AutoTagApply.SegmentsToToggles` relies on it only loosely (it re-sorts).

Video frame `k` shows clip frame `startFrame + k * frameStep`, with
`frameStep = max(1, round(clipFrameRate / renderFps))`. The video is encoded at `renderFps` even
when that rounding makes playback slightly off real time (24 fps clips) — the burned-in counter,
not the video clock, is the source of truth.

## Annotate CLI and stdout protocol

`python -m autotag.annotate <run> [--model ID] [--only GUID ...] [--resume]`, cwd `Python/`.
Unity launches it via `AutoTagAnnotateProcess` with the venv from `PythonRuntime.ResolveVenv()`,
`PYTHONUNBUFFERED=1`, and `GEMINI_API_KEY` copied from the process or user registry environment.

Lines Unity parses (anything else is just logged): `PROGRESS <i> <n> <name>`,
`USAGE <guid> <in> <out>`, `CLIPERROR <guid> <msg>`, `DONE <ok> <failed>`. Exit codes: 0 all ok,
2 some clips failed (results still written, apply proceeds), 1 fatal. argparse's own exit 2 is
overridden to 1 so it cannot be mistaken for partial success.

`USAGE` is printed only for real calls, not for `--resume` re-parses, so cost is never counted
twice. Output tokens include thinking tokens because those are billed.

## Per-tag requests

`perTagRequests` in the manifest (the window's *One Request Per Tag*, off by default) makes
`annotate` send one request per tag, each with a prompt and schema naming only that tag, over a
single upload of the video. The responses' `tags` lists are merged before post-processing, so
`results.json` looks the same either way. Every request re-bills the video's input tokens, which is
why it is off by default and why the window's estimate multiplies by the tag count.

Whichever mode, the prompt states that tags are independent and may overlap. Without that sentence
the model treated tags as exclusive classes: the first trial produced zero frames carrying two tags.

## Hazards

- **`--only` rewrites `results.json` with just the selected clips.** Applying after an `--only`
  run touches only those clips; it does not merge with a previous full run.
- **Only `autotag.gemini` imports `google.genai`**, and `annotate` imports it lazily. The test
  suite asserts the SDK is never loaded, so the stdlib-only suite runs without it. Keep it that way.
- **The Gemini calls use the Interactions API** (`client.interactions.create`, video item field
  `resolution`, `processing: {type: static, fps}`, `response_format.schema`). These were verified
  against google-genai 2.25.0 by introspection, not by a live call; its connection error type is
  imported from a private module because it has no public path.
- **Rendering uses `Hidden/MoSynth/AutoTagShaded`**, an unlit shader that shades from the camera
  side and draws the checker floor itself, because URP preview-scene lighting was not dependable.
  4x MSAA on the render target produced a white frame and a URP "Missing resolve surface" error,
  so anti-aliasing is 2x supersampling plus a downscaling blit instead.
- **The frame label is drawn onto the read-back pixels** (`BurnInLabel`, a 5x7 bitmap font), not as
  scene geometry, so camera changes can never move or hide it.
- **Left/right colouring comes from `BoneNameConventions.SideOf`.** A rig whose bone names carry
  no side marker renders all grey, and the model loses its left/right cue.
- **Apply skips channels that already hold keys** unless the user picks Overwrite; with Overwrite,
  a tag the model found nowhere clears the existing channel. A tag with no segments never creates
  a channel, and a clip never gains an empty `AnimationTagComponent`.
- **Assembly reload is locked while annotate runs**, and closing the window kills the process.
