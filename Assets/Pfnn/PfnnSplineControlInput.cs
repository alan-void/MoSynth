using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace Pfnn
{
/// <summary>
/// Follows a spline at a constant speed, handing the network the stretch of path ahead of the
/// character as its future trajectory.
/// </summary>
/// <remarks>
/// Unlike <c>MotionFieldSplineControlInput</c>, which steers by pure pursuit toward a single
/// lookahead point, a PFNN wants the whole future window — so this samples the path at each of the
/// horizons the network asks for. It advances on its own clock rather than tracking where the
/// character actually is, which is what makes drift show up as measurable path-following error
/// instead of being steered out; that is the same choice <c>SplineControlInput</c> makes, and what
/// makes both usable as measurement tools.
/// </remarks>
public class PfnnSplineControlInput : PfnnControlInput, IMotionSynthesisSplineControlInput
{
    [Tooltip("Path to follow. Closed splines loop; on open splines the character stops at the end.")]
    [SerializeField]
    private SplineContainer splineContainer;

    [Tooltip("Travel speed along the spline, in m/s.")]
    [SerializeField]
    private float speed = 1.0f;

    public SplineContainer SplineContainer
    {
        get => splineContainer;
        set
        {
            splineContainer = value;
            // A benchmark sweep hands this component a different spline for every run, and the
            // tracked position means nothing on the new one.
            _splineT = 0f;
        }
    }

    public float TargetSpeed => speed;

    /// <summary>The direction the path sets off in; a bare path carries no facing of its own.</summary>
    public float3 GetWorldInitDirection() => SplinePathDirection.WorldStartDirection(splineContainer);

    /// <summary>Position along the path, normalized to 0..1.</summary>
    private float _splineT;

    private bool HasPath => splineContainer != null && splineContainer.Spline != null;
    private bool IsClosed => HasPath && splineContainer.Spline.Closed;
    private float PathLength => HasPath ? splineContainer.CalculateLength() : 0f;

    /// <summary>Seconds of path a single synthesis frame covers, as a fraction of the whole.</summary>
    private float _normalizedSpeedPerFrame;

    protected override void OnUpdate()
    {
        var pathLength = PathLength;
        if (pathLength <= 1e-5f) return;

        var frameTime = 1f / math.max(1f, Synthesizer.synthesisFrameRate);
        _normalizedSpeedPerFrame = speed / pathLength * frameTime;

        _splineT = SplineFold.Normalized(_splineT + speed / pathLength * Time.deltaTime, IsClosed);
    }

    public override bool TryGetFutureSample(int frameOffset, out float2 position,
        out float2 direction)
    {
        position = default;
        direction = default;
        if (!HasPath || _normalizedSpeedPerFrame <= 0f) return false;

        var t = SplineFold.Normalized(_splineT + frameOffset * _normalizedSpeedPerFrame, IsClosed);
        var sampled = splineContainer.EvaluatePosition(t);
        position = new float2(sampled.x, sampled.z);

        // Measured as a short step further along rather than as the analytic tangent, so the facing
        // handed to the network cannot disagree with the positions beside it.
        var aheadT = SplineFold.Normalized(t + _normalizedSpeedPerFrame, IsClosed);
        var ahead = splineContainer.EvaluatePosition(aheadT);
        var step = new float2(ahead.x - sampled.x, ahead.z - sampled.z);

        // At the clamped end of an open path the step forward lands on the same point, so measure
        // the last one instead of falling through to an arbitrary default.
        if (math.lengthsq(step) < 1e-12f)
        {
            var behindT = SplineFold.Normalized(t - _normalizedSpeedPerFrame, IsClosed);
            var behind = splineContainer.EvaluatePosition(behindT);
            step = new float2(sampled.x - behind.x, sampled.z - behind.z);
        }

        direction = math.normalizesafe(step, new float2(0f, 1f));
        return true;
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying || Stage == null) return;

        Gizmos.color = new Color(0.2f, 0.8f, 1f);
        for (var frames = 0; frames <= 30; frames += 5)
        {
            if (!TryGetFutureSample(frames, out var position, out var direction)) continue;
            var world = new Vector3(position.x, RootPosition.y + 0.05f, position.y);
            Gizmos.DrawSphere(world, 0.04f);
            Gizmos.DrawLine(world, world + new Vector3(direction.x, 0f, direction.y) * 0.2f);
        }
    }
#endif
}
}
