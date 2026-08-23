using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace AnimationTools
{
/// <summary>
/// Watches one benchmark run and decides when it is over: added to the spawned character, it counts
/// the character's own progress round the path and reports completion or a timeout.
/// </summary>
/// <remarks>
/// Progress is only accumulated once <see cref="settleTime"/> has elapsed, so the measured lap is a
/// lap of settled locomotion rather than of the character scrambling away from its spawn pose. The
/// nearest-point query runs at the default pick resolution and outside the pipeline's timed region,
/// so it does not show up in the cost metrics.
/// </remarks>
[DefaultExecutionOrder(1001)]
public class BenchmarkLapProbe : MonoBehaviour
{
    public MotionSynthesisComponent synthesizer;
    public SplineContainer spline;

    [Min(0f)] public float settleTime = 3f;
    [Min(0.1f)] public float lapsRequired = 1f;
    [Min(1f)] public float maxRunSeconds = 120f;

    /// <summary>True once the required laps are done or the time limit is hit. The driver polls this.</summary>
    public bool IsFinished { get; private set; }

    /// <summary>True when <see cref="IsFinished"/> was reached by running out of time rather than by finishing.</summary>
    public bool TimedOut { get; private set; }

    /// <summary>Laps completed since the settle time elapsed. Clamped at zero; a character that drifted backwards reports 0.</summary>
    public float CompletedLaps => _tracker == null ? 0f : math.max(0f, _tracker.Progress);

    /// <summary>Seconds of synthesis time since the run began, settle time included.</summary>
    public float ElapsedSeconds { get; private set; }

    public int TickCount { get; private set; }

    private SplineLapTracker _tracker;
    private readonly SplineProjector _projector = new();
    private bool _settled;
    private bool _running;

    /// <summary>Starts watching. Call after the character is active and its synthesizer has woken up.</summary>
    public void Begin()
    {
        if (_running) return;

        if (synthesizer == null || spline == null || spline.Spline == null)
        {
            Debug.LogError($"{nameof(BenchmarkLapProbe)} on \"{name}\": synthesizer or spline missing; the run cannot be timed.", this);
            IsFinished = true;
            TimedOut = true;
            return;
        }

        _tracker = new SplineLapTracker(spline.Spline.Closed);
        _projector.Reset();
        _settled = settleTime <= 0f;
        ElapsedSeconds = 0f;
        TickCount = 0;
        IsFinished = false;
        TimedOut = false;

        synthesizer.OnPoseApplied += HandlePoseApplied;
        _running = true;
    }

    /// <summary>Stops watching. Safe to call more than once.</summary>
    public void End()
    {
        if (!_running) return;
        if (synthesizer != null) synthesizer.OnPoseApplied -= HandlePoseApplied;
        _running = false;
    }

    private void HandlePoseApplied(PoseBuffer pose, float deltaTime)
    {
        ElapsedSeconds += deltaTime;
        TickCount++;

        if (!_settled)
        {
            if (ElapsedSeconds < settleTime)
            {
                CheckTimeout();
                return;
            }

            // Re-anchor exactly at the settle boundary so the first measured lap starts here.
            _settled = true;
            _tracker.Reset();
        }

        _tracker.Sample(NearestT());

        if (_tracker.HasCompleted(lapsRequired))
        {
            IsFinished = true;
            return;
        }

        CheckTimeout();
    }

    private void CheckTimeout()
    {
        if (ElapsedSeconds < maxRunSeconds) return;
        IsFinished = true;
        TimedOut = true;
    }

    /// <summary>
    /// Where on the path the character currently is, as a normalized parameter. The projection works
    /// in the spline's local space, so the root is pulled through the container transform first —
    /// containers are assumed unscaled, as everywhere else that touches these splines.
    /// </summary>
    /// <remarks>
    /// Projected with continuity rather than by global proximity, because <see cref="SplineLapTracker"/>
    /// reads a large parameter jump as a seam crossing. On a path that crosses itself the globally
    /// nearest point flips branches, and a lap would be counted from a jump the character never made.
    /// </remarks>
    private float NearestT()
    {
        var root = synthesizer.transform.position;
        var localRoot = (float3)spline.transform.InverseTransformPoint(root);
        return _projector.Project(spline.Spline, localRoot);
    }

    private void OnDisable()
    {
        End();
    }
}
}
