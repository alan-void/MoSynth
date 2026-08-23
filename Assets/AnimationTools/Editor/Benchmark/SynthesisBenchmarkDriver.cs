using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Splines;
using Object = UnityEngine.Object;

namespace AnimationTools.Editor
{
/// <summary>
/// Runs a whole benchmark sweep inside one play session: every configured method against every
/// path, one character alive at a time, writing a report when the queue empties.
/// </summary>
/// <remarks>
/// The queue lives on the Editor side and is pumped from <see cref="EditorApplication.update"/>
/// rather than from a MonoBehaviour, for two reasons. Loading path and character prefabs needs
/// <see cref="AssetDatabase"/>, which has no business in the runtime assembly; and driving the
/// sweep from outside the scene means no state has to survive between runs. The only runtime piece
/// is <see cref="BenchmarkLapProbe"/>, which answers the one question the Editor cannot: has this
/// character been round the path yet.
/// <para>
/// Runs never overlap. Two characters sharing a frame would contend for the same cores and the same
/// Python GIL, and the cost metrics would then measure that contention rather than either method.
/// </para>
/// </remarks>
[InitializeOnLoad]
public static class SynthesisBenchmarkDriver
{
    private enum Phase
    {
        WaitingForPlayMode,
        Spawn,
        Running,
        Teardown,
        Cooldown,
        Complete
    }

    /// <summary>Editor ticks to idle between destroying one character and spawning the next.</summary>
    private const int CooldownTicks = 2;

    /// <summary>
    /// Ticks to wait for play mode after arming. A sweep that never gets there — a compile error, a
    /// scene that refuses to load — has to fail loudly rather than hang a headless run forever.
    /// </summary>
    private const int PlayModeTimeoutTicks = 600;

    private static bool _armed;
    private static Phase _phase;

    private static SynthesisBenchmarkPlan _plan;
    private static SynthesisBenchmarkConfig _config;
    private static List<GameObject> _pathPrefabs;
    private static readonly List<BenchmarkRunResult> Results = new();

    private static int _pathIndex;
    private static int _methodIndex;
    private static int _cooldown;
    private static int _waitedForPlayMode;
    private static int _restoreVSyncCount = -1;

    private static GameObject _spawnHolder;
    private static GameObject _pathInstance;
    private static SplineContainer _spline;
    private static GameObject _character;
    private static MotionRecorder _recorder;
    private static BenchmarkLapProbe _probe;
    private static StageCostChannel _costChannel;
    private static string _recordingsDirectory;

    static SynthesisBenchmarkDriver()
    {
        // Re-arms on the far side of the play-mode domain reload. Harmless on every other reload:
        // without a plan file there is nothing to do.
        if (File.Exists(SynthesisBenchmarkPlan.PlanPath)) Arm();
    }

    /// <summary>
    /// Starts watching for play mode so the sweep begins as soon as it starts. Idempotent, and safe
    /// to call either side of a domain reload — whichever call survives is the one that runs.
    /// </summary>
    public static void Arm()
    {
        if (_armed) return;
        _armed = true;
        _phase = Phase.WaitingForPlayMode;
        _waitedForPlayMode = 0;
        EditorApplication.update += Tick;
    }

    private static void Disarm()
    {
        if (!_armed) return;
        _armed = false;
        EditorApplication.update -= Tick;
    }

    private static void Tick()
    {
        try
        {
            Step();
        }
        catch (Exception e)
        {
            Debug.LogError($"[Benchmark] Sweep aborted: {e}");
            Finish(success: false);
        }
    }

    private static void Step()
    {
        // Play mode ending under the sweep takes the scene with it. Every later run would then spawn
        // into nothing and record a destroyed-object error, so stop at the first sign of it and keep
        // the runs that did finish.
        if (_phase != Phase.WaitingForPlayMode && _phase != Phase.Complete && !EditorApplication.isPlaying)
        {
            Debug.LogError($"[Benchmark] Play mode ended after {Results.Count} run(s); the sweep was cut short.");
            Finish(success: false);
            return;
        }

        switch (_phase)
        {
            case Phase.WaitingForPlayMode:
                if (!EditorApplication.isPlaying)
                {
                    if (++_waitedForPlayMode < PlayModeTimeoutTicks) return;
                    Debug.LogError("[Benchmark] Play mode never started; the sweep cannot run.");
                    Finish(success: false);
                    return;
                }

                Setup();
                _phase = Phase.Spawn;
                return;

            case Phase.Spawn:
                Spawn();
                return;

            case Phase.Running:
                if (_probe != null && !_probe.IsFinished) return;
                _phase = Phase.Teardown;
                return;

            case Phase.Teardown:
                Teardown();
                _cooldown = CooldownTicks;
                _phase = Phase.Cooldown;
                return;

            case Phase.Cooldown:
                if (--_cooldown > 0) return;
                _methodIndex++;
                _phase = Phase.Spawn;
                return;

            case Phase.Complete:
                Disarm();
                return;
        }
    }

    private static void Setup()
    {
        _plan = JsonUtility.FromJson<SynthesisBenchmarkPlan>(File.ReadAllText(SynthesisBenchmarkPlan.PlanPath));

        // Deleted as soon as it has been read: from here the sweep lives in memory, and a leftover
        // file would restart it on the next domain reload.
        File.Delete(SynthesisBenchmarkPlan.PlanPath);

        _config = AssetDatabase.LoadAssetAtPath<SynthesisBenchmarkConfig>(_plan.configAssetPath);
        if (_config == null)
            throw new InvalidOperationException($"No SynthesisBenchmarkConfig at \"{_plan.configAssetPath}\".");

        if (!_config.TryValidate(out var error))
            throw new InvalidOperationException($"Config \"{_plan.configAssetPath}\" is not runnable: {error}");

        _pathPrefabs = ResolvePathPrefabs(_config);
        if (_pathPrefabs.Count == 0)
            throw new InvalidOperationException("No usable path prefabs; every candidate lacked a SplineContainer.");

        _recordingsDirectory = Path.Combine(_plan.outputDirectory, "recordings");
        Directory.CreateDirectory(_recordingsDirectory);

        // An unfocused Editor otherwise throttles to a few ticks a second, which would stretch a
        // headless sweep out to nothing useful.
        Application.runInBackground = true;

        if (_config.fixedTimestep && _config.synthesisFrameRate > 1e-5f)
        {
            // Decouples the run from the wall clock: every rendered frame advances exactly one
            // synthesis tick, so the sweep runs as fast as the CPU allows and produces the same
            // tick count every time.
            Time.captureDeltaTime = 1f / _config.synthesisFrameRate;

            // With frames no longer tied to real time, vsync is the only thing left that would
            // still pace them — and it would silently undo the fast-forward in a windowed Editor.
            _restoreVSyncCount = QualitySettings.vSyncCount;
            QualitySettings.vSyncCount = 0;
        }

        _pathIndex = 0;
        _methodIndex = 0;
        Results.Clear();

        _spawnHolder = new GameObject("BenchmarkSpawnHolder");
        _spawnHolder.SetActive(false);

        Debug.Log($"[Benchmark] Sweep starting: {_config.methods.Count} method(s) x {_pathPrefabs.Count} path(s) " +
                  $"= {_config.methods.Count * _pathPrefabs.Count} runs -> \"{_plan.outputDirectory}\".");
    }

    private static void Spawn()
    {
        if (_pathIndex >= _pathPrefabs.Count)
        {
            Finish(success: true);
            return;
        }

        if (_methodIndex >= _config.methods.Count)
        {
            // This path is done. Advancing the index here, rather than inferring "new path" from a
            // null instance, keeps a vanished instance from silently restarting the method list.
            if (_pathInstance != null) Object.DestroyImmediate(_pathInstance);
            _pathInstance = null;
            _spline = null;
            _pathIndex++;
            _methodIndex = 0;
            return;
        }

        if (_pathInstance == null)
        {
            var prefab = _pathPrefabs[_pathIndex];
            _pathInstance = Object.Instantiate(prefab);
            _pathInstance.name = prefab.name;
            _spline = _pathInstance.GetComponentInChildren<SplineContainer>(true);
        }

        StartRun();
    }

    private static void StartRun()
    {
        var method = _config.methods[_methodIndex];
        var pathName = _pathPrefabs[_pathIndex].name;
        var totalRuns = _config.methods.Count * _pathPrefabs.Count;

        try
        {
            // Instantiated under an inactive holder so Awake does not run until the character has
            // been placed and its overrides applied. MotionSynthesisComponent seeds its pose from
            // the transforms in Awake, so a character that wakes at the origin and is moved
            // afterwards starts with a bogus root velocity.
            _character = Object.Instantiate(method.characterPrefab, _spawnHolder.transform);
            _character.name = method.name;

            var synthesizer = _character.GetComponentInChildren<MotionSynthesisComponent>(true);
            if (synthesizer == null)
                throw new InvalidOperationException($"\"{method.characterPrefab.name}\" has no MotionSynthesisComponent.");

            PlaceAtPathStart(_character.transform, synthesizer.transform, _spline);

            var input = FindSplineControlInput(_character);
            if (input == null)
                throw new InvalidOperationException(
                    $"\"{method.characterPrefab.name}\" has no component implementing IMotionSynthesisSplineControlInput.");
            input.SplineContainer = _spline;

            if (method.overrides != null)
            {
                foreach (var methodOverride in method.overrides)
                {
                    methodOverride?.Apply(_character);
                }
            }

            if (!HasOverride<SynthesisFrameRateOverride>(method))
                synthesizer.synthesisFrameRate = _config.synthesisFrameRate;

            // Reparenting to the scene root activates the character, which is what finally runs
            // Awake — by now against the placed transform.
            _character.transform.SetParent(null, worldPositionStays: true);

            if (!synthesizer.isActiveAndEnabled || synthesizer.PoseLayout == null)
                throw new InvalidOperationException(
                    "the synthesis component disabled itself during Awake (see the errors above).");

            AttachRecorder(synthesizer, method, pathName);
            AttachProbe(synthesizer);

            Debug.Log($"[Benchmark] Run {Results.Count + 1}/{totalRuns}: {method.DescribeFull()} on {pathName}.");
            _phase = Phase.Running;
        }
        catch (Exception e)
        {
            Debug.LogError($"[Benchmark] {method.name} on {pathName} could not start: {e.Message}");

            var failed = new BenchmarkRunResult
            {
                method = method.name,
                path = pathName,
                error = e.Message
            };
            Results.Add(failed);
            AppendProgress(failed);

            DestroyCharacter();
            _cooldown = CooldownTicks;
            _phase = Phase.Cooldown;
        }
    }

    private static void AttachRecorder(MotionSynthesisComponent synthesizer, BenchmarkMethod method, string pathName)
    {
        _costChannel = new StageCostChannel { name = "cost" };

        _recorder = _character.AddComponent<MotionRecorder>();
        _recorder.autoStartOnPlay = false;
        _recorder.Synthesizer = synthesizer;
        _recorder.trigger = RecordTrigger.EverySynthesisUpdate;
        _recorder.outputDirectory = _recordingsDirectory;
        _recorder.recordingName = Sanitize($"{method.name}_{pathName}");
        _recorder.channels = new List<RecorderChannel>
        {
            new TimeChannel { name = "time" },
            new BoneWorldPositionChannel { name = "root", simulationBone = true },
            new BoneWorldForwardChannel { name = "rootForward", simulationBone = true },
            new ContactBoneWorldPositionChannel { name = "footL", left = true },
            new ContactBoneWorldPositionChannel { name = "footR", left = false },
            new FootContactChannel { name = "contacts" },
            new PoseDiscontinuityChannel { name = "discontinuity" },
            _costChannel
        };

        if (_config.recordFullPose) _recorder.channels.Add(new FullPoseChannel { name = "pose" });

        _recorder.StartRecording();
        if (!_recorder.IsRecording)
            throw new InvalidOperationException("the recorder refused to start (see the errors above).");
    }

    private static void AttachProbe(MotionSynthesisComponent synthesizer)
    {
        _probe = _character.AddComponent<BenchmarkLapProbe>();
        _probe.synthesizer = synthesizer;
        _probe.spline = _spline;
        _probe.settleTime = _config.settleTime;
        _probe.lapsRequired = _config.lapsRequired;
        _probe.maxRunSeconds = _config.maxRunSeconds;
        _probe.Begin();
    }

    private static void Teardown()
    {
        var method = _config.methods[_methodIndex];
        var pathName = _pathPrefabs[_pathIndex].name;

        if (_recorder == null || _probe == null)
        {
            // The run never really happened — recording an empty row would put a zero into the
            // report as if it were a measurement.
            Debug.LogError($"[Benchmark] {method.name} on {pathName} lost its recorder or probe; no row written.");
            DestroyCharacter();
            return;
        }

        var result = new BenchmarkRunResult
        {
            method = method.name,
            path = pathName,
            timedOut = _probe != null && _probe.TimedOut,
            completedLaps = _probe != null ? _probe.CompletedLaps : 0f,
            durationSeconds = _probe != null ? _probe.ElapsedSeconds : 0f
        };

        // Read before the character is destroyed: the target speed lives on its control input.
        var targetSpeed = FindTargetSpeed();

        _probe?.End();
        _recorder?.StopRecording();
        _costChannel?.DisposeRecorder();

        try
        {
            if (_recorder != null)
            {
                result.recordingFile = Path.GetFileName(_recorder.ManifestPath);
                BenchmarkRunEvaluator.Evaluate(_recorder.ManifestPath, _config, targetSpeed, _spline, result);
            }
        }
        catch (Exception e)
        {
            result.error = $"evaluation failed: {e.Message}";
            Debug.LogError($"[Benchmark] {method.name} on {pathName}: {e}");
        }

        if (result.timedOut)
        {
            Debug.LogWarning($"[Benchmark] {method.name} on {pathName} timed out after {result.durationSeconds:0.0} s, " +
                             $"having covered {result.completedLaps:0.00} of {_config.lapsRequired:0.##} lap(s).");
        }

        Results.Add(result);
        AppendProgress(result);
        DestroyCharacter();
    }

    private static float FindTargetSpeed()
    {
        var input = _character != null ? FindSplineControlInput(_character) : null;
        return input?.TargetSpeed ?? float.NaN;
    }

    private static void DestroyCharacter()
    {
        // Immediate rather than deferred, so stage teardown — which for the motion field releases
        // Python state — has finished before the next run's cost is measured.
        if (_character != null) Object.DestroyImmediate(_character);
        _character = null;
        _recorder = null;
        _probe = null;
        _costChannel = null;
        GC.Collect();
    }

    private static void Finish(bool success)
    {
        _phase = Phase.Complete;
        Time.captureDeltaTime = 0f;
        if (_restoreVSyncCount >= 0)
        {
            QualitySettings.vSyncCount = _restoreVSyncCount;
            _restoreVSyncCount = -1;
        }

        if (_pathInstance != null) Object.DestroyImmediate(_pathInstance);
        if (_spawnHolder != null) Object.DestroyImmediate(_spawnHolder);
        _pathInstance = null;
        _spawnHolder = null;

        // Written whenever there is anything to write, not only on success: a sweep that was cut
        // short still produced real numbers for the runs that completed, and throwing those away
        // helps nobody. Success governs the exit code alone.
        if (_config != null && _plan != null && Results.Count > 0)
        {
            try
            {
                var csvPath = BenchmarkReportWriter.Write(_plan.outputDirectory, _config, _plan.configAssetPath, Results);
                Debug.Log($"[Benchmark] {(success ? "Sweep complete" : "Partial results")}: " +
                          $"{Results.Count} run(s) -> \"{csvPath}\".");
            }
            catch (Exception e)
            {
                Debug.LogError($"[Benchmark] Report could not be written: {e}");
                success = false;
            }
        }

        var exitEditor = _plan != null && _plan.exitEditorWhenDone;
        Disarm();

        if (exitEditor)
        {
            EditorApplication.Exit(success ? 0 : 1);
            return;
        }

        if (EditorApplication.isPlaying) EditorApplication.ExitPlaymode();
    }

    /// <summary>
    /// Appends one finished run to a JSON-lines file as it completes, so a sweep that is killed half
    /// way still leaves usable numbers behind. The real report is written at the end.
    /// </summary>
    private static void AppendProgress(BenchmarkRunResult result)
    {
        try
        {
            File.AppendAllText(Path.Combine(_plan.outputDirectory, "progress.jsonl"),
                JsonUtility.ToJson(result) + "\n");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Benchmark] Could not append progress: {e.Message}");
        }
    }

    /// <summary>
    /// Puts the character on the first point of the path, facing along it. Starting anywhere else
    /// spends the settle time walking to the path, which makes the first lap a measure of the
    /// approach rather than of the following.
    /// </summary>
    /// <param name="character">The spawned prefab root, which is what actually gets moved.</param>
    /// <param name="characterFrame">
    /// The transform that has to end up on the path. This is the synthesis component's own
    /// transform, not the prefab root — it is what root motion drives and what every measurement
    /// reads, and in these character prefabs it sits about two metres off the root.
    /// </param>
    public static void PlaceAtPathStart(Transform character, Transform characterFrame, SplineContainer container)
    {
        var spline = container.Spline;
        var splineTransform = container.transform;

        var worldPosition = splineTransform.TransformPoint((Vector3)spline.EvaluatePosition(0f));

        var forward = splineTransform.TransformDirection((Vector3)spline.EvaluateTangent(0f));
        forward.y = 0f;

        if (forward.sqrMagnitude <= 1e-8f)
        {
            // A knot authored with a linear tangent evaluates to a zero tangent, so a path with
            // square corners has no analytic direction at its start. A short chord along the path
            // does, and agrees with the tangent everywhere the tangent exists.
            var length = spline.GetLength();
            var step = length > 1e-3f ? Mathf.Min(0.1f, length * 0.05f) / length : 0f;
            forward = splineTransform.TransformPoint((Vector3)spline.EvaluatePosition(step)) - worldPosition;
            forward.y = 0f;
        }

        var rotation = forward.sqrMagnitude > 1e-8f
            ? Quaternion.LookRotation(forward.normalized, Vector3.up)
            : characterFrame.rotation;

        // Move the root by whatever delta lands the character frame on the path, so the character's
        // internal offsets survive. Rotation first: the translation is measured after it, against
        // where the frame has ended up.
        var deltaRotation = rotation * Quaternion.Inverse(characterFrame.rotation);
        character.rotation = deltaRotation * character.rotation;
        character.position += worldPosition - characterFrame.position;
    }

    private static IMotionSynthesisSplineControlInput FindSplineControlInput(GameObject root)
    {
        foreach (var behaviour in root.GetComponentsInChildren<MonoBehaviour>(true))
        {
            if (behaviour is IMotionSynthesisSplineControlInput input) return input;
        }

        return null;
    }

    private static bool HasOverride<T>(BenchmarkMethod method) where T : BenchmarkOverride
    {
        if (method.overrides == null) return false;

        foreach (var methodOverride in method.overrides)
        {
            if (methodOverride is T) return true;
        }

        return false;
    }

    /// <summary>
    /// The path prefabs this sweep will run, in a stable order. The folder wins over the explicit
    /// list when set, so adding a path is a matter of dropping a prefab in rather than editing the
    /// config; candidates without a SplineContainer are reported and skipped.
    /// </summary>
    private static List<GameObject> ResolvePathPrefabs(SynthesisBenchmarkConfig config)
    {
        var candidates = new List<GameObject>();

        if (!string.IsNullOrWhiteSpace(config.pathPrefabFolder) && AssetDatabase.IsValidFolder(config.pathPrefabFolder))
        {
            var guids = AssetDatabase.FindAssets("t:Prefab", new[] { config.pathPrefabFolder });
            var paths = new List<string>(guids.Length);
            foreach (var guid in guids) paths.Add(AssetDatabase.GUIDToAssetPath(guid));
            paths.Sort(StringComparer.Ordinal);

            foreach (var path in paths)
            {
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (prefab != null) candidates.Add(prefab);
            }
        }
        else
        {
            if (!string.IsNullOrWhiteSpace(config.pathPrefabFolder))
            {
                Debug.LogWarning($"[Benchmark] \"{config.pathPrefabFolder}\" is not a folder; " +
                                 "falling back to the explicit path list.");
            }

            foreach (var prefab in config.pathPrefabs)
            {
                if (prefab != null) candidates.Add(prefab);
            }
        }

        var usable = new List<GameObject>(candidates.Count);
        foreach (var prefab in candidates)
        {
            if (prefab.GetComponentInChildren<SplineContainer>(true) != null) usable.Add(prefab);
            else Debug.LogWarning($"[Benchmark] Skipping \"{prefab.name}\": no SplineContainer.");
        }

        return usable;
    }

    /// <summary>Makes a method/path pair safe to use as a recording filename.</summary>
    private static string Sanitize(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = name.ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (Array.IndexOf(invalid, chars[i]) >= 0 || chars[i] == ' ') chars[i] = '_';
        }

        return new string(chars);
    }
}
}
