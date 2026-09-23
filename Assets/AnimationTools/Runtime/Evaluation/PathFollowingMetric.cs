using System;
using System.Collections.Generic;
using System.Text;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Splines;

namespace AnimationTools
{
/// <summary>
/// Benchmark harness for path following: pushes one spline into a set of spline control inputs,
/// records each character's run with a MotionRecorder, then computes and logs trajectory error
/// (mean distance to the spline), heading error against the spline tangent, and velocity error
/// against the input's target speed. Results are also shown read-only in the inspector.
/// The spline container is assumed unscaled.
/// </summary>
public class PathFollowingMetric : MonoBehaviour
{
    [Tooltip("The path to follow. Pushed into every listed control input at startup, and the reference " +
             "the metrics are computed against. Its transform must be unscaled.")]
    [SerializeField] private SplineContainer spline;

    [Tooltip("The control inputs to benchmark (e.g. SplineControlInput, MotionFieldSplineControlInput). " +
             "Each gets the spline above assigned and its character recorded and evaluated separately.")]
    [SerializeField] private List<MonoBehaviour> splineControlInputs = new();

    [Tooltip("Seconds to run before evaluating. 0 = run until StopAndEvaluate() is called.")]
    [SerializeField, Min(0f)] private float duration = 20f;

    [Tooltip("Initial seconds excluded from the metrics while the character reaches the path.")]
    [SerializeField, Min(0f)] private float settleTime = 2f;

    [Tooltip("Where the .bin/.json recordings are written. Relative paths resolve against the " +
             "project root, next to Assets.")]
    [SerializeField] private string outputDirectory = "Recordings";

    [Tooltip("Also record each character's full pose buffer every tick, for offline analysis beyond " +
             "these metrics. Not needed for the metrics themselves.")]
    [SerializeField] private bool recordFullPose;

    [Tooltip("Width in seconds of the moving average applied to the finite-differenced speed before " +
             "computing velocity error. Raw per-tick speed is dominated by synthesis jitter (frame " +
             "switches, gait-cycle sway), which would inflate the error. 0 disables smoothing.")]
    [SerializeField, Min(0f)] private float speedSmoothingWindow = 0.2f;

    [Tooltip("Metrics from the last completed run, one entry per control input. Read-only display.")]
    [SerializeField, InspectorReadOnly] private List<PathFollowingMetricsResult> results = new();

    /// <summary>True between a successful startup and the run being stopped/evaluated.</summary>
    public bool IsRunning { get; private set; }

    /// <summary>Metrics from the last completed run, one entry per control input.</summary>
    public IReadOnlyList<PathFollowingMetricsResult> Results => results;

    private class Entry
    {
        public IMotionSynthesisSplineControlInput Input;
        public MonoBehaviour Behaviour;
        public MotionRecorder Recorder;
    }

    private readonly List<Entry> _entries = new();
    private float _startTime;

    private void OnValidate()
    {
        for (var i = 0; i < splineControlInputs.Count; i++)
        {
            var entry = splineControlInputs[i];
            if (entry == null) continue;
            if (entry is IMotionSynthesisSplineControlInput) continue;

            Debug.LogError($"PathFollowingMetric \"{name}\": {entry.GetType().Name} on \"{entry.name}\" does not implement IMotionSynthesisSplineControlInput.", this);
            splineControlInputs[i] = null;
        }
    }

    private void Start()
    {
        if (spline == null)
        {
            Debug.LogError($"PathFollowingMetric \"{name}\": no spline assigned.", this);
            enabled = false;
            return;
        }

        var validInputs = new List<IMotionSynthesisSplineControlInput>();
        foreach (var behaviour in splineControlInputs)
        {
            var input = behaviour as IMotionSynthesisSplineControlInput;
            if (input == null) continue;

            if (input.Synthesizer == null)
            {
                Debug.LogError($"PathFollowingMetric \"{name}\": {input.GetType().Name} has no Synthesizer assigned.", this);
                continue;
            }

            validInputs.Add(input);
        }

        if (validInputs.Count == 0)
        {
            Debug.LogError($"PathFollowingMetric \"{name}\": no valid spline control inputs configured.", this);
            enabled = false;
            return;
        }

        for (var i = 0; i < validInputs.Count; i++)
        {
            var input = validInputs[i];
            var behaviour = input as MonoBehaviour;
            Assert.IsNotNull(behaviour, $"PathFollowingMetric \"{name}\": {input.GetType().Name} is not a MonoBehaviour.");
            input.SplineContainer = spline;

            var recorder = gameObject.AddComponent<MotionRecorder>();
            recorder.autoStartOnPlay = false;
            recorder.Synthesizer = input.Synthesizer;
            recorder.trigger = RecordTrigger.EverySynthesisUpdate;
            recorder.outputDirectory = outputDirectory;
            recorder.recordingName = $"pathmetric_{behaviour.gameObject.name}_{i}";
            recorder.channels = new List<RecorderChannel>
            {
                new TimeChannel { name = "time" },
                new BoneWorldPositionChannel { name = "root", simulationBone = true },
                new BoneWorldForwardChannel { name = "rootForward", simulationBone = true }
            };
            if (recordFullPose) recorder.channels.Add(new FullPoseChannel { name = "pose" });

            recorder.StartRecording();
            if (!recorder.IsRecording)
            {
                Debug.LogError($"PathFollowingMetric \"{name}\": failed to start recording for \"{behaviour.gameObject.name}\".", this);
                Destroy(recorder);
                continue;
            }

            _entries.Add(new Entry { Input = input, Behaviour = behaviour, Recorder = recorder });
        }

        if (_entries.Count == 0)
        {
            Debug.LogError($"PathFollowingMetric \"{name}\": no recordings could be started.", this);
            enabled = false;
            return;
        }

        IsRunning = true;
        _startTime = Time.time;
        results.Clear();
    }

    private void Update()
    {
        if (IsRunning && duration > 0f && Time.time - _startTime >= duration) StopAndEvaluate();
    }

    /// <summary>
    /// Stops all recorders, reads each recording back from disk, computes the metrics, logs one
    /// summary block per character and fills <see cref="Results"/>. Called automatically when
    /// <see cref="duration"/> elapses; call it manually when duration is 0.
    /// </summary>
    public void StopAndEvaluate()
    {
        if (!IsRunning) return;
        IsRunning = false;

        foreach (var entry in _entries)
        {
            entry.Recorder.StopRecording();

            var label = $"{entry.Behaviour.gameObject.name} ({entry.Behaviour.GetType().Name})";
            try
            {
                var reader = RecordingReader.Load(entry.Recorder.ManifestPath);

                var timeChannel = reader.GetChannel("time");
                var rootChannel = reader.GetChannel("root");
                var forwardChannel = reader.GetChannel("rootForward");

                var times = new float[reader.FrameCount];
                var positions = new float3[reader.FrameCount];
                var forwards = new float3[reader.FrameCount];
                for (var f = 0; f < reader.FrameCount; f++)
                {
                    times[f] = reader.GetFloat(f, timeChannel, 0);
                    positions[f] = reader.GetFloat3(f, rootChannel);
                    forwards[f] = reader.GetFloat3(f, forwardChannel);
                }

                var result = PathFollowingMetricsCalculator.Evaluate(
                    spline.Spline,
                    (float4x4)spline.transform.localToWorldMatrix,
                    positions,
                    forwards,
                    times,
                    entry.Input.TargetSpeed,
                    settleTime,
                    speedSmoothingWindow);
                result.label = label;
                results.Add(result);

                if (result.framesEvaluated == 0)
                    Debug.LogWarning($"[PathFollowingMetric] {result.label}: not enough frames after settle time to evaluate.", this);
                else
                    Debug.Log(Format(result), this);
            }
            catch (Exception e)
            {
                Debug.LogError($"PathFollowingMetric: evaluation failed for {label}: {e}");
            }

            Destroy(entry.Recorder);
        }

        _entries.Clear();
    }

    private string Format(PathFollowingMetricsResult r)
    {
        var sb = new StringBuilder();
        sb.Append($"[PathFollowingMetric] {r.label}: {r.framesEvaluated} frames evaluated (settle {settleTime:0.0} s)\n");

        sb.Append($"  Trajectory error   mean {r.meanTrajectoryError:0.000} m\n");

        sb.Append($"  Heading error      mean {r.meanHeadingErrorDeg:0.0} deg, max {r.maxHeadingErrorDeg:0.0} deg\n");

        sb.Append(float.IsNaN(r.targetSpeed)
            ? $"  Speed              actual {r.meanActualSpeed:0.00} m/s, target n/a (no speed model)"
            : $"  Speed              actual {r.meanActualSpeed:0.00} m/s, target {r.targetSpeed:0.00} m/s, mean abs error {r.meanVelocityError:0.00} m/s");

        return sb.ToString();
    }

    private void OnDisable()
    {
        if (!IsRunning) return;
        IsRunning = false;
        foreach (var entry in _entries) entry.Recorder.StopRecording();
        _entries.Clear();
    }
}
}
