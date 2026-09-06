---
type: Agent Reference
title: Verifying a change
description: How to compile, test, and regenerate databases in this project from outside the Editor, and the two ways that loop silently lies to you.
tags: [agents, tooling, unity, tests, workflow]
sources:
  - id: openwiki-source-7fb5a5b0459d70a3874a6002
    resource: repo://Assets/AnimationTools/Tests/Editor/RunTestsMenu.cs
  - id: openwiki-source-94ccd2c9da4f13a61c72d13f
    resource: repo://Assets/AnimationTools/Tests/Editor/TestResultDump.cs
  - id: openwiki-source-9fdf1041c917e3aac18aa9f8
    resource: repo://Assets/MotionMatching/Editor/Core/MotionMatchingDatabaseMenu.cs
  - id: openwiki-source-28684d61d3dab2bdc41d31f4
    resource: repo://Assets/MotionMatching/Editor/Core/MotionMatchingDataEditor.cs
  - id: openwiki-source-030d30d689203655d06f8a6b
    resource: repo://Python/tests/test_gait_phase.py
  - id: openwiki-source-44aeec7103fbb90f34a34626
    resource: repo://Python/tests/test_training_data.py
generated: {by: "claude-code", at: "2026-08-30T16:03:03.865Z"}
---

# Verifying a change

The Unity Editor is usually already open on this project, and driving it through the Unity MCP tools
is the whole verification loop: there is no headless build step and no `dotnet build` that produces
the same assemblies.

## Compiling

`Assets/Refresh` (menu item) triggers the reimport and recompile. Nothing else is needed.

> **Compile errors do not appear in the Editor console read.** `Unity_ReadConsole` with `Types:
> ["Error"]` returns nothing at all for a C# compile failure. Unity keeps the last good DLL and the
> stale assembly keeps answering, so everything downstream — including a test run — looks like it
> succeeded against your change when it ran against the previous one.

Read `%LOCALAPPDATA%/Unity/Editor/Editor.log` instead. The reliable shape is: note the line count
before refreshing, then poll the tail for either signal.

```bash
LOG="$LOCALAPPDATA/Unity/Editor/Editor.log"
MARK=$(wc -l < "$LOG")          # take this BEFORE triggering Assets/Refresh
# ... trigger Assets/Refresh ...
for i in $(seq 1 25); do
  sleep 3
  tail -n +$MARK "$LOG" | grep -q "error CS" && { tail -n +$MARK "$LOG" | grep "error CS" | sort -u; break; }
  tail -n +$MARK "$LOG" | grep -q "Begin MonoManager ReloadAssembly" && { echo "reloaded, no errors"; break; }
done
```

`Begin MonoManager ReloadAssembly` is the domain reload, which only happens after a successful
compile. Waiting for one of those two lines is what stops the next step running against stale
assemblies. Cross-check `Library/ScriptAssemblies/<Assembly>.dll` timestamps if you are unsure which
generation is loaded.

## Running the edit-mode tests

`MoSynth/Tests/Run EditMode Tests` runs `AnimationTools.Tests`, `MotionMatching.Tests` and
`Pfnn.Tests`.

That list is hardcoded as `Filter.assemblyNames` in
`Assets/AnimationTools/Tests/Editor/TestResultDump.cs`. **A new test assembly must be added to it or
its tests silently do not run** — the count comes back unchanged and nothing says why. The same list
is why an assembly that is deleted has to be removed from it.

The run spans a domain reload, so results cannot come back through the menu call. `TestResultDump`
writes them to `Temp/animtools_test_results.txt` instead:

```
passed=236 failed=0 skipped=0
FAIL <full test name>: <message>
```

Delete that file before the run and wait for it to reappear — otherwise you read the previous run's
summary and cannot tell.

**Do not trust a passing run you did not first confirm compiled.** The two failure modes compose: a
compile error leaves the old assemblies loaded, the test count comes back looking plausible, and the
number is from code you did not write.

## Regenerating the databases

`MoSynth/Database/Regenerate Motion Matching Databases` rebuilds every `MotionMatchingData` asset's
`.mmpose` and `.mmfeatures`. Run it after any change to an extraction format or a feature definition.

`Assets/StreamingAssets/` is gitignored, so generated databases are local artefacts — regenerating is
always safe and never shows up in a diff. There is no byte-parity constraint to preserve.

## Running the Python tests

```bash
python -m unittest discover -s Python/tests -t Python/tests
```

The `-t Python` form fails with *"Start directory is not importable"* — there is no
`Python/tests/__init__.py`, so the tests directory has to be its own top level. Each test file puts
`Python/` on `sys.path` itself, which is why that works.

The suites use only numpy and scipy and build their own fixtures, so they need neither Unity nor a
generated database. Use the project venv (see [running the Python side](../python/running-the-python-side.md)).

## An OpenWiki run overwrites part of AGENTS.md

The lifecycle rewrites the block between the `<!-- OPENWIKI:START -->` / `<!-- OPENWIKI:END -->`
markers in root `AGENTS.md` with its own default text. That block currently holds the project's
two-audience wiki policy — including the statement that this `agents/` section belongs to the coding
agents — and the default text replaces it with the opposite advice.

**`openwiki_begin` does this, not only `openwiki_finish`.** A backup copied after `begin` is already
the clobbered version, so it is useless as a restore source and a post-`finish` diff against it looks
clean.

**Check `git diff AGENTS.md` after both calls, and restore with `git show HEAD:AGENTS.md`.** The
`wikiGoal` returned by `openwiki_begin` also still carries the real policy in full.

## Confirming it actually runs

`Assets/Scenes/ExampleSimpleMMStages.unity` is the demo scene, and entering play mode for a few
seconds and capturing the scene view is a cheap end-to-end check that the pipeline still produces a
pose. Stop play mode afterwards; a play session leaves the scene clean if you change nothing.
