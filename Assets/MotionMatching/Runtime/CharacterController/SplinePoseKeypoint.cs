using System;
using System.Collections.Generic;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace MotionMatching
{
/// <summary>
/// Builds a spline from a <see cref="SkeletonAnimation"/>'s root trajectory and remembers which pose
/// the clip held at a configurable frame interval along it. Each keypoint stores its clip frame, its
/// normalized spline parameter, and the character's facing there; a bone's constraint target is that
/// frame's bone position expressed relative to the keypoint's own simulation frame, re-anchored onto
/// the spline's frame at the keypoint's parameter.
/// </summary>
/// <remarks>
/// <para>
/// Facing is stored explicitly rather than derived from the path tangent, because the two are not the
/// same thing: the bone offsets are expressed in the clip's simulation frame, which is facing-aligned,
/// so re-anchoring a strafing or backpedalling pose against the direction of travel would place the
/// feet rotated about the character. It is stored as a signed yaw <em>offset</em> from the tangent, so
/// the generated path reproduces the clip's facing exactly while dragging knots into a new path keeps
/// the relationship — a 30° strafe stays a 30° strafe relative to the new heading.
/// </para>
/// <para>
/// Knot density is independent of keypoint spacing: knots are fitted to
/// <see cref="simplificationToleranceMeters"/> so the curve stays sparse enough to edit, and each
/// keypoint is located on it by projection. The GameObject's transform places the path in the world
/// and must be unscaled and upright — the spline convention across the project, and what makes the
/// stored yaw offset valid against the world tangent.
/// </para>
/// </remarks>
public class SplinePoseKeypoint : MonoBehaviour
{
    [Tooltip("The animation whose root trajectory becomes the spline and whose frames become the keypoint poses.")]
    public SkeletonAnimation skeletonAnimation;

    [Min(1)]
    [Tooltip("A pose keypoint is stored every this many clip frames. Independent of knot placement.")]
    public int keypointIntervalFrames = 20;

    [Min(0.001f)]
    [Tooltip("Knot-placement control: the root path is simplified until it deviates by about this much, " +
             "so knots cluster through turns and thin out on straights. Larger means a sparser, more " +
             "editable curve that tracks the original path more loosely.")]
    public float simplificationToleranceMeters = 0.1f;

    [Tooltip("Close the spline into a loop. Leave off for non-looping clips; note the control input " +
             "wraps its parameter, so an open spline snaps back to the start when it ends.")]
    public bool closed;

    [SerializeField] private SplineContainer splineContainer;
    [SerializeField] private List<Keypoint> keypoints = new();

    [Serializable]
    public struct Keypoint
    {
        [Tooltip("Frame of the source animation this keypoint's pose comes from.")]
        public int frameIndex;

        [Tooltip("Normalized spline parameter where this keypoint sits.")]
        public float splineT;

        [Tooltip("Signed yaw in radians from the path tangent to the character's facing.")]
        public float facingYawOffset;
    }

    public SplineContainer SplineContainer => splineContainer;
    public Skeleton Skeleton => skeletonAnimation != null ? skeletonAnimation.Skeleton : null;
    public IReadOnlyList<Keypoint> Keypoints => keypoints;

    /// <summary>Number of knots in the fitted curve; 0 before <see cref="Rebuild"/>.</summary>
    public int KnotCount => splineContainer != null && splineContainer.Spline != null ? splineContainer.Spline.Count : 0;

    // AnnotatedAnimationClip shadows FrameCount/GetFrame with its startFrame/endFrame window, so a
    // base-typed access here would silently sample outside the slice.
    private int ClipFrameCount =>
        skeletonAnimation is AnnotatedAnimationClip annotated ? annotated.FrameCount : skeletonAnimation.FrameCount;

    private PoseBuffer GetClipFrame(int frameIndex) =>
        skeletonAnimation is AnnotatedAnimationClip annotated
            ? annotated.GetFrame(frameIndex)
            : skeletonAnimation.GetFrame(frameIndex);

    /// <summary>
    /// Regenerates the spline and the keypoint list from the assigned animation. Knots are fitted to
    /// the clip's root path in this GameObject's local space, so the transform places the whole path.
    /// </summary>
    [ContextMenu("Rebuild Spline And Keypoints")]
    public void Rebuild()
    {
        keypoints.Clear();

        if (skeletonAnimation == null)
        {
            Debug.LogError("[SplinePoseKeypoint] Cannot rebuild: no SkeletonAnimation assigned.", this);
            return;
        }
        if (!skeletonAnimation.TryValidate(out var error))
        {
            Debug.LogError($"[SplinePoseKeypoint] Cannot rebuild: {error}", this);
            return;
        }
        if (transform.lossyScale != Vector3.one)
        {
            Debug.LogWarning("[SplinePoseKeypoint] The transform is scaled; splines are assumed unscaled, " +
                             "so keypoint targets will be wrong.", this);
        }

        var skeleton = skeletonAnimation.Skeleton;
        var skeletonData = skeleton.GetSkeletonData();
        var def = SimulationFrameDef.Default(skeleton);
        var frameCount = ClipFrameCount;

        // The clip's root path, one sample per frame: positions fit the curve, rotations give facing.
        var pathPositions = new float3[frameCount];
        var pathFacings = new float3[frameCount];
        for (var frame = 0; frame < frameCount; frame++)
        {
            SimulationFrame.Compute(GetClipFrame(frame), skeletonData, def, out var framePos, out var frameRot);
            pathPositions[frame] = framePos; // already ground-projected (y = 0)
            pathFacings[frame] = math.mul(frameRot, math.forward());
        }

        var knotFrames = Simplify(pathPositions, simplificationToleranceMeters);
        if (knotFrames.Count < 2)
        {
            Debug.LogError("[SplinePoseKeypoint] The root path collapsed to fewer than two knots; the clip " +
                           "may not move, or the tolerance may be larger than the path.", this);
            return;
        }

        var spline = new Spline();
        foreach (var frame in knotFrames) spline.Add(new BezierKnot(pathPositions[frame]), TangentMode.AutoSmooth);
        spline.Closed = closed;
        spline.Warmup(); // curve-length and up-vector LUTs, before the projector hits them

        if (splineContainer == null && !TryGetComponent(out splineContainer))
        {
            splineContainer = gameObject.AddComponent<SplineContainer>();
        }
        splineContainer.Spline = spline;

        // Keypoints keep their own cadence, and are placed by arc length rather than by projecting
        // them onto the curve: nearest-point projection is ambiguous wherever a path crosses itself,
        // and the correspondence is already known — the curve passes exactly through the knots, so a
        // keypoint interpolates between its bracketing knots' distances. Monotonic by construction.
        var cumulative = CumulativeLengths(pathPositions);
        var knotDistances = KnotDistances(spline, knotFrames.Count);
        var totalLength = spline.GetLength();
        for (var frame = 0; frame < frameCount; frame += keypointIntervalFrames)
        {
            var splineT = totalLength > 1e-5f
                ? DistanceAtFrame(frame, knotFrames, knotDistances, cumulative) / totalLength
                : 0f;

            keypoints.Add(new Keypoint
            {
                frameIndex = frame,
                splineT = splineT,
                facingYawOffset = SignedYawAngle(LocalTangent(spline, splineT), pathFacings[frame])
            });
        }

#if UNITY_EDITOR
        UnityEditor.EditorUtility.SetDirty(this);
        UnityEditor.EditorUtility.SetDirty(splineContainer);
#endif
    }

    /// <summary>Distance travelled along the sampled path up to each frame.</summary>
    private static float[] CumulativeLengths(IReadOnlyList<float3> points)
    {
        var cumulative = new float[points.Count];
        for (var i = 1; i < points.Count; i++)
        {
            cumulative[i] = cumulative[i - 1] + math.distance(points[i], points[i - 1]);
        }
        return cumulative;
    }

    /// <summary>Distance along the spline to each knot. Curve <c>j</c> runs from knot <c>j</c> to
    /// knot <c>j+1</c>, so knot <c>j</c>'s distance is the sum of the curves before it.</summary>
    private static float[] KnotDistances(Spline spline, int knotCount)
    {
        var distances = new float[knotCount];
        for (var j = 1; j < knotCount; j++)
        {
            distances[j] = distances[j - 1] + spline.GetCurveLength(j - 1);
        }
        return distances;
    }

    /// <summary>
    /// Where a clip frame sits along the spline, in metres. Exact at a knot; between knots the frame's
    /// share of the sampled path's length carries over to the curve's.
    /// </summary>
    private static float DistanceAtFrame(int frame, List<int> knotFrames, float[] knotDistances, float[] cumulative)
    {
        // The last knot at or before this frame. Knot frames ascend, and RDP keeps the final sample,
        // so a keypoint frame always lands inside the range.
        var knot = 0;
        for (var j = 0; j < knotFrames.Count; j++)
        {
            if (knotFrames[j] > frame) break;
            knot = j;
        }
        if (knot >= knotFrames.Count - 1) return knotDistances[^1];

        var fromFrame = knotFrames[knot];
        var toFrame = knotFrames[knot + 1];
        var span = cumulative[toFrame] - cumulative[fromFrame];
        var u = span > 1e-6f ? (cumulative[frame] - cumulative[fromFrame]) / span : 0f;
        return math.lerp(knotDistances[knot], knotDistances[knot + 1], math.saturate(u));
    }

    /// <summary>
    /// Douglas-Peucker over the XZ path: keeps the samples needed to stay within
    /// <paramref name="tolerance"/> of the original polyline, so knots follow the path's shape rather
    /// than the clock. Returned indices are ascending and always include the first and last sample.
    /// </summary>
    /// <remarks>
    /// The retained samples become AutoSmooth knots, and the curve then bows <em>through</em> them, so
    /// the fitted spline deviates somewhat more than the tolerance. It is a knot-placement control,
    /// not a hard error bound.
    /// </remarks>
    private static List<int> Simplify(IReadOnlyList<float3> points, float tolerance)
    {
        var keep = new bool[points.Count];
        if (points.Count == 0) return new List<int>();
        keep[0] = true;
        keep[points.Count - 1] = true;

        var toleranceSq = tolerance * tolerance;
        var pending = new Stack<(int First, int Last)>();
        pending.Push((0, points.Count - 1));
        while (pending.Count > 0)
        {
            var (first, last) = pending.Pop();
            if (last <= first + 1) continue;

            var worst = -1;
            var worstDistanceSq = toleranceSq;
            for (var i = first + 1; i < last; i++)
            {
                var distanceSq = DistanceSqToSegment(points[i], points[first], points[last]);
                if (distanceSq <= worstDistanceSq) continue;
                worst = i;
                worstDistanceSq = distanceSq;
            }
            if (worst < 0) continue;

            keep[worst] = true;
            pending.Push((first, worst));
            pending.Push((worst, last));
        }

        var indices = new List<int>();
        for (var i = 0; i < keep.Length; i++)
        {
            if (keep[i]) indices.Add(i);
        }
        return indices;
    }

    private static float DistanceSqToSegment(float3 point, float3 start, float3 end)
    {
        var segment = end - start;
        var lengthSq = math.lengthsq(segment);
        var closest = lengthSq < 1e-12f
            ? start
            : start + segment * math.saturate(math.dot(point - start, segment) / lengthSq);
        return math.lengthsq(point - closest);
    }

    /// <summary>
    /// Nearest keypoint to a normalized spline parameter, wrap-aware when the spline is closed.
    /// False when no keypoint lies within <paramref name="tolerance"/> (also normalized).
    /// </summary>
    public bool TryGetKeypointNearT(float t, float tolerance, out Keypoint keypoint)
    {
        keypoint = default;
        var best = float.MaxValue;
        foreach (var candidate in keypoints)
        {
            var delta = math.abs(t - candidate.splineT);
            if (closed) delta = math.min(delta, 1f - delta);
            if (delta < best)
            {
                best = delta;
                keypoint = candidate;
            }
        }
        return best <= tolerance;
    }

    /// <summary>
    /// World position a bone should reach at a keypoint: the keypoint pose's bone offset, taken
    /// relative to its own simulation frame, re-anchored onto the spline's frame at the keypoint's
    /// parameter. Y is carried through as height above the spline's plane.
    /// </summary>
    public bool TryGetWorldBonePosition(in Keypoint keypoint, int boneIndex, out float3 world)
    {
        world = float3.zero;
        if (splineContainer == null || skeletonAnimation == null || !skeletonAnimation.TryValidate(out _)) return false;

        var skeleton = skeletonAnimation.Skeleton;
        var skeletonData = skeleton.GetSkeletonData();
        if (boneIndex < 0 || boneIndex >= skeletonData.BoneCount) return false;

        var pose = GetClipFrame(keypoint.frameIndex);
        SimulationFrame.Compute(pose, skeletonData, SimulationFrameDef.Default(skeleton),
            out var clipFramePos, out var clipFrameRot);
        var local = SimulationFrame.ToFrameLocal(skeletonData.CharacterSpacePosition(pose, boneIndex),
            clipFramePos, clipFrameRot);

        GetWorldFrame(keypoint.splineT, out var splinePos, out var splineRot);
        world = SimulationFrame.FromFrameLocal(local, splinePos, splineRot);
        return true;
    }

    /// <summary>The frame a character following this path should be in at a parameter: the spline's
    /// position, and the facing interpolated from the keypoints rather than the path tangent.</summary>
    public void GetWorldFrame(float t, out float3 position, out quaternion rotation)
    {
        position = splineContainer.EvaluatePosition(t);
        rotation = GetWorldFacingRotation(t);
    }

    /// <summary>Yaw-only rotation aiming along the facing this path carries at a parameter.</summary>
    public quaternion GetWorldFacingRotation(float t) =>
        quaternion.LookRotationSafe(GetWorldFacing(t), math.up());

    /// <summary>
    /// The facing direction at a parameter: the world path tangent, rotated by the yaw offset
    /// interpolated between the bracketing keypoints. Falls back to the tangent with no keypoints.
    /// </summary>
    public float3 GetWorldFacing(float t)
    {
        var tangent = WorldTangent(t);
        var offset = GetFacingYawOffset(t);
        return math.mul(quaternion.AxisAngle(math.up(), offset), tangent);
    }

    /// <summary>The raw path direction at a parameter, ignoring facing. Diagnostic and gizmo use.</summary>
    public float3 GetWorldTangent(float t) => WorldTangent(t);

    /// <summary>
    /// Yaw offset from tangent to facing at a parameter, lerped along the shortest arc between the
    /// bracketing keypoints. Exact at a keypoint's own parameter.
    /// </summary>
    private float GetFacingYawOffset(float t)
    {
        if (keypoints.Count == 0) return 0f;
        if (keypoints.Count == 1) return keypoints[0].facingYawOffset;

        // Keypoints are in ascending parameter order, so the bracket is the last one at or before t.
        var previous = -1;
        for (var i = 0; i < keypoints.Count; i++)
        {
            if (keypoints[i].splineT > t) break;
            previous = i;
        }

        if (previous < 0) return closed ? LerpAcross(keypoints.Count - 1, 0, t) : keypoints[0].facingYawOffset;
        if (previous == keypoints.Count - 1)
        {
            return closed ? LerpAcross(keypoints.Count - 1, 0, t) : keypoints[^1].facingYawOffset;
        }
        return LerpAcross(previous, previous + 1, t);
    }

    /// <summary>Interpolates the yaw offset between two keypoints, handling the wrap of a closed
    /// spline's final span and the wrap of the angles themselves.</summary>
    private float LerpAcross(int fromIndex, int toIndex, float t)
    {
        var from = keypoints[fromIndex];
        var to = keypoints[toIndex];

        var span = to.splineT - from.splineT;
        var offsetIntoSpan = t - from.splineT;
        if (span <= 0f) // the closing span of a loop, running from the last keypoint back to the first
        {
            span += 1f;
            if (offsetIntoSpan < 0f) offsetIntoSpan += 1f;
        }

        var u = span > 1e-6f ? math.saturate(offsetIntoSpan / span) : 0f;
        return from.facingYawOffset + WrapPi(to.facingYawOffset - from.facingYawOffset) * u;
    }

    private float3 WorldTangent(float t) => Flatten(splineContainer.EvaluateTangent(t));

    private static float3 LocalTangent(Spline spline, float t) => Flatten(spline.EvaluateTangent(t));

    /// <summary>Flattened onto the ground plane and normalized; EvaluateTangent is neither.</summary>
    private static float3 Flatten(float3 direction) =>
        math.normalizesafe(new float3(direction.x, 0f, direction.z), math.forward());

    /// <summary>Signed angle in radians from one ground-plane direction to another, about +Y.</summary>
    private static float SignedYawAngle(float3 from, float3 to) =>
        math.atan2(math.dot(math.cross(from, to), math.up()), math.dot(from, to));

    /// <summary>An angle folded into (-pi, pi], so interpolation takes the short way round.</summary>
    private static float WrapPi(float radians)
    {
        radians = (radians + math.PI) % (2f * math.PI);
        if (radians < 0f) radians += 2f * math.PI;
        return radians - math.PI;
    }

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        if (splineContainer == null || keypoints.Count == 0) return;
        if (skeletonAnimation == null || !skeletonAnimation.TryValidate(out _)) return;

        var skeleton = skeletonAnimation.Skeleton;
        var hasLeft = BoneNameConventions.TryFindContactBone(skeleton, left: true, out var leftIndex);
        var hasRight = BoneNameConventions.TryFindContactBone(skeleton, left: false, out var rightIndex);

        foreach (var keypoint in keypoints)
        {
            var position = (float3)splineContainer.EvaluatePosition(keypoint.splineT);

            // Facing bright, the path's own direction dim: where they diverge is where storing facing
            // separately earns its keep.
            Gizmos.color = new Color(0.35f, 0.35f, 0.35f);
            var tangent = GetWorldTangent(keypoint.splineT);
            GizmosExtensions.DrawArrow(position, position + tangent * 0.3f, 0.08f, thickness: 2);

            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(position, 0.05f);
            var facing = GetWorldFacing(keypoint.splineT);
            GizmosExtensions.DrawArrow(position, position + facing * 0.4f, 0.1f, thickness: 3);

            Gizmos.color = Color.cyan;
            if (hasLeft && TryGetWorldBonePosition(keypoint, leftIndex, out var leftPos))
            {
                Gizmos.DrawSphere(leftPos, 0.03f);
            }
            if (hasRight && TryGetWorldBonePosition(keypoint, rightIndex, out var rightPos))
            {
                Gizmos.DrawSphere(rightPos, 0.03f);
            }
        }
    }
#endif
}
}
