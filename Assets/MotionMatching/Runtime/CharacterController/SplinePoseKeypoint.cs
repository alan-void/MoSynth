using System;
using System.Collections.Generic;
using AnimationTools;
using Unity.Collections;
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
/// a keypoint reads as "30° off the heading here" rather than as a bare world direction.
/// </para>
/// <para>
/// Knot density is independent of keypoint spacing: knots are fitted to
/// <see cref="simplificationToleranceMeters"/>, so they cluster through turns and thin out on
/// straights. The GameObject's transform places the path in the world and must be unscaled and
/// upright — the spline convention across the project, and what makes the stored yaw offset valid
/// against the world tangent.
/// </para>
/// <para>
/// Both the path and the keypoints are generated here and held un-serialized, rather than the path
/// living in a <c>SplineContainer</c> and the keypoints in the scene. They are built together from the
/// clip in one pass, so they cannot describe different curves, and there is no editable copy for
/// anyone to drag out of agreement — an emptied container used to send <c>speed / length</c> to
/// infinity and a NaN parameter into every evaluation downstream. The path is therefore exactly the
/// clip's root trajectory; to place it differently, move or rotate the GameObject.
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

    [Tooltip("Index of the keypoint whose pose is drawn as a skeleton in the Scene view; -1 draws none. " +
             "One at a time on purpose: a skeleton at every keypoint is thousands of gizmo lines per repaint.")]
    public int drawKeypointSkeletonAtIndex = -1;

    [NonSerialized] private Spline _path;
    [NonSerialized] private List<Keypoint> _keypoints;

    // What the cache was built from. OnValidate does not fire for edits made to the animation *asset*,
    // so the clip's frame count is part of the fingerprint rather than trusting the callback alone.
    [NonSerialized] private SkeletonAnimation _builtFrom;
    [NonSerialized] private int _builtFrameCount;
    [NonSerialized] private int _builtInterval;
    [NonSerialized] private float _builtTolerance;
    [NonSerialized] private bool _builtClosed;

    public struct Keypoint
    {
        /// <summary>Frame of the source animation this keypoint's pose comes from.</summary>
        public int frameIndex;

        /// <summary>Normalized spline parameter where this keypoint sits.</summary>
        public float splineT;

        /// <summary>Signed yaw in radians from the path tangent to the character's facing.</summary>
        public float facingYawOffset;
    }

    public Skeleton Skeleton => skeletonAnimation != null ? skeletonAnimation.Skeleton : null;

    /// <summary>The keypoints, built from the clip on first access. Never null; empty when the clip
    /// cannot be read.</summary>
    public IReadOnlyList<Keypoint> Keypoints => Keys;

    /// <summary>The keypoint cache, via the one access point that guarantees it has been built.</summary>
    private List<Keypoint> Keys
    {
        get
        {
            EnsureBuilt();
            return _keypoints;
        }
    }

    /// <summary>The generated path, via the same access point, so it and the keypoints are never
    /// out of step. Null when the clip cannot be read.</summary>
    private Spline Path
    {
        get
        {
            EnsureBuilt();
            return _path;
        }
    }

    /// <summary>Whether there is a usable path. Everything that evaluates the curve must check this
    /// first — a path of fewer than two knots has no length to divide by.</summary>
    public bool HasPath
    {
        get
        {
            var path = Path;
            return path != null && path.Count >= 2;
        }
    }

    /// <summary>Number of knots in the fitted curve; 0 when the clip cannot be read.</summary>
    public int KnotCount => Path?.Count ?? 0;

    /// <summary>The path's length in world units, or 0 when there is no path.</summary>
    public float WorldLength
    {
        get
        {
            var path = Path;
            return path != null ? SplineUtility.CalculateLength(path, transform.localToWorldMatrix) : 0f;
        }
    }

    /// <summary>Position on the path at a normalized parameter, in world space.</summary>
    /// <remarks>
    /// The spline is fitted in this GameObject's local space, so the transform maps it to the world —
    /// the same evaluate-then-transform that <c>SplineContainer</c> does for an unscaled transform,
    /// which this class already requires.
    /// </remarks>
    public float3 EvaluateWorldPosition(float t)
    {
        var path = Path;
        return path != null
            ? math.transform(transform.localToWorldMatrix, path.EvaluatePosition(t))
            : (float3)transform.position;
    }

    /// <summary>Path direction at a normalized parameter, in world space. Not normalized, and zero
    /// where the curve has no analytic tangent.</summary>
    public float3 EvaluateWorldTangent(float t)
    {
        var path = Path;
        return path != null
            ? math.mul((float4x4)transform.localToWorldMatrix, new float4(path.EvaluateTangent(t), 0f)).xyz
            : float3.zero;
    }

    // AnnotatedAnimationClip shadows FrameCount/GetFrame with its startFrame/endFrame window, so a
    // base-typed access here would silently sample outside the slice.
    private int ClipFrameCount =>
        skeletonAnimation is AnnotatedAnimationClip annotated ? annotated.FrameCount : skeletonAnimation.FrameCount;

    private PoseBuffer GetClipFrame(int frameIndex) =>
        skeletonAnimation is AnnotatedAnimationClip annotated
            ? annotated.GetFrame(frameIndex)
            : skeletonAnimation.GetFrame(frameIndex);

    /// <summary>
    /// Regenerates the path and the keypoints now, and says so when it cannot. Nothing here is
    /// serialized, so this exists for the report: the lazy path builds silently on demand and would
    /// otherwise leave a misconfigured clip looking merely empty.
    /// </summary>
    [ContextMenu("Rebuild Spline And Keypoints")]
    public void Rebuild()
    {
        if (transform.lossyScale != Vector3.one)
        {
            Debug.LogWarning("[SplinePoseKeypoint] The transform is scaled; splines are assumed unscaled, " +
                             "so keypoint targets will be wrong.", this);
        }

        if (!TryBuildFromClip(out var path, out var built, out var error))
        {
            _path = null;
            _keypoints = new List<Keypoint>();
            RecordCacheInputs();
            Debug.LogError($"[SplinePoseKeypoint] Cannot rebuild: {error}", this);
            return;
        }

        _path = path;
        _keypoints = built;
        RecordCacheInputs();
        Debug.Log($"[SplinePoseKeypoint] Rebuilt: {built.Count} keypoints over {path.Count} knots.", this);
    }

    /// <summary>
    /// Builds the keypoints, and the spline they are measured against, from the clip alone. That
    /// spline is fitted here rather than read from <see cref="splineContainer"/> on purpose — see the
    /// class remarks. Reports failure instead of logging it, so the lazy path can call it on a repaint.
    /// </summary>
    private bool TryBuildFromClip(out Spline fittedSpline, out List<Keypoint> keypoints, out string error)
    {
        fittedSpline = null;
        keypoints = null;

        if (skeletonAnimation == null)
        {
            error = "no SkeletonAnimation assigned.";
            return false;
        }
        if (!skeletonAnimation.TryValidate(out error)) return false;

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
            error = "the root path collapsed to fewer than two knots; the clip may not move, or the " +
                    "tolerance may be larger than the path.";
            return false;
        }

        var spline = new Spline();
        foreach (var frame in knotFrames) spline.Add(new BezierKnot(pathPositions[frame]), TangentMode.AutoSmooth);
        spline.Closed = closed;
        spline.Warmup(); // curve-length and up-vector LUTs, before the projector hits them

        // Keypoints keep their own cadence, and are placed by arc length rather than by projecting
        // them onto the curve: nearest-point projection is ambiguous wherever a path crosses itself,
        // and the correspondence is already known — the curve passes exactly through the knots, so a
        // keypoint interpolates between its bracketing knots' distances. Monotonic by construction.
        var cumulative = CumulativeLengths(pathPositions);
        var knotDistances = KnotDistances(spline, knotFrames.Count);
        var totalLength = spline.GetLength();
        keypoints = new List<Keypoint>();
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

        fittedSpline = spline;
        return true;
    }

    /// <summary>
    /// Builds the path and the keypoints if they are missing or were built from different inputs. A
    /// clip that cannot be read caches an empty result rather than retrying, so a repainting gizmo
    /// does not re-attempt the build every frame; changing any input gets it rebuilt.
    /// </summary>
    private void EnsureBuilt()
    {
        if (_keypoints != null && IsCacheCurrent()) return;

        if (TryBuildFromClip(out var path, out var built, out _))
        {
            _path = path;
            _keypoints = built;
        }
        else
        {
            _path = null;
            _keypoints = new List<Keypoint>();
        }
        RecordCacheInputs();
    }

    private bool IsCacheCurrent() =>
        _builtFrom == skeletonAnimation &&
        _builtFrameCount == CachedFrameCount &&
        _builtInterval == keypointIntervalFrames &&
        _builtTolerance == simplificationToleranceMeters &&
        _builtClosed == closed;

    private void RecordCacheInputs()
    {
        _builtFrom = skeletonAnimation;
        _builtFrameCount = CachedFrameCount;
        _builtInterval = keypointIntervalFrames;
        _builtTolerance = simplificationToleranceMeters;
        _builtClosed = closed;
    }

    /// <summary>The clip's frame count, or 0 with no clip. Part of the cache fingerprint, because
    /// widening the clip's window changes every keypoint without touching this component.</summary>
    private int CachedFrameCount => skeletonAnimation != null ? ClipFrameCount : 0;

    private void OnValidate()
    {
        _keypoints = null;
        _path = null;
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
        var found = false;
        var best = float.MaxValue;
        foreach (var candidate in Keys)
        {
            var delta = math.abs(t - candidate.splineT);
            if (closed) delta = math.min(delta, 1f - delta);
            if (delta >= best) continue;
            best = delta;
            keypoint = candidate;
            found = true;
        }
        // Tested separately from the distance: a NaN t never beats float.MaxValue, and an infinite
        // tolerance would otherwise pass that unset distance and hand back a zeroed keypoint.
        return found && best <= tolerance;
    }

    /// <summary>
    /// World position a bone should reach at a keypoint: the keypoint pose's bone offset, taken
    /// relative to its own simulation frame, re-anchored onto the spline's frame at the keypoint's
    /// parameter. Y is carried through as height above the spline's plane.
    /// </summary>
    public bool TryGetWorldBonePosition(in Keypoint keypoint, int boneIndex, out float3 world)
    {
        world = float3.zero;
        if (!TryGetAnchor(keypoint, out var skeletonData, out var pose, out var anchor)) return false;
        if (boneIndex < 0 || boneIndex >= skeletonData.BoneCount) return false;

        world = anchor.Apply(skeletonData.CharacterSpacePosition(pose, boneIndex));
        return true;
    }

    /// <summary>
    /// World positions of every bone at a keypoint, re-anchored onto the spline exactly as
    /// <see cref="TryGetWorldBonePosition"/> re-anchors one. Solves the pose in a single forward pass
    /// rather than walking a parent chain per bone, which is what makes drawing a whole skeleton
    /// affordable. <paramref name="world"/> must hold at least the skeleton's bone count.
    /// </summary>
    public bool TryGetWorldBonePositions(in Keypoint keypoint, NativeArray<float3> world)
    {
        if (!TryGetAnchor(keypoint, out var skeletonData, out var pose, out var anchor)) return false;
        if (world.Length < skeletonData.BoneCount) return false;
        // The clip's bake and the rig are invalidated by different things, so they can end up
        // describing two different skeletons. Answer nothing rather than let the FK pass index one
        // by the other's bone count.
        if (pose.Layout.RotationCount != skeletonData.BoneCount) return false;

        using var positions = new NativeArray<float3>(skeletonData.BoneCount, Allocator.Temp);
        using var rotations = new NativeArray<quaternion>(skeletonData.BoneCount, Allocator.Temp);
        skeletonData.LocalSpaceToCharacterSpace(pose, positions, rotations);

        for (var i = 0; i < skeletonData.BoneCount; i++) world[i] = anchor.Apply(positions[i]);
        return true;
    }

    /// <summary>
    /// A keypoint's pose and the map that carries it onto the path. Shared by the single-bone and
    /// whole-skeleton lookups so they cannot drift apart on what "re-anchored onto the spline" means.
    /// </summary>
    private bool TryGetAnchor(in Keypoint keypoint, out SkeletonData skeletonData, out PoseBuffer pose,
        out KeypointAnchor anchor)
    {
        skeletonData = default;
        pose = default;
        anchor = default;
        if (!HasPath || skeletonAnimation == null || !skeletonAnimation.TryValidate(out _)) return false;

        var skeleton = skeletonAnimation.Skeleton;
        skeletonData = skeleton.GetSkeletonData();
        pose = GetClipFrame(keypoint.frameIndex);

        SimulationFrame.Compute(pose, skeletonData, SimulationFrameDef.Default(skeleton),
            out var clipFramePos, out var clipFrameRot);
        GetWorldFrame(keypoint.splineT, out var splinePos, out var splineRot);
        anchor = new KeypointAnchor(clipFramePos, clipFrameRot, splinePos, splineRot);
        return true;
    }

    /// <summary>
    /// The rigid map from a keypoint pose's own simulation frame onto the spline's frame at that
    /// keypoint. Held as a value so a whole skeleton costs one frame solve rather than one per bone.
    /// </summary>
    private readonly struct KeypointAnchor
    {
        private readonly float3 _clipPosition;
        private readonly quaternion _clipRotation;
        private readonly float3 _splinePosition;
        private readonly quaternion _splineRotation;

        public KeypointAnchor(float3 clipPosition, quaternion clipRotation,
            float3 splinePosition, quaternion splineRotation)
        {
            _clipPosition = clipPosition;
            _clipRotation = clipRotation;
            _splinePosition = splinePosition;
            _splineRotation = splineRotation;
        }

        /// <summary>Carries a character-space bone position from the keypoint's clip frame onto the
        /// path. Y comes through as height above the spline's plane.</summary>
        public float3 Apply(float3 characterSpacePosition) =>
            SimulationFrame.FromFrameLocal(
                SimulationFrame.ToFrameLocal(characterSpacePosition, _clipPosition, _clipRotation),
                _splinePosition, _splineRotation);
    }

    /// <summary>The frame a character following this path should be in at a parameter: the spline's
    /// position, and the facing interpolated from the keypoints rather than the path tangent.</summary>
    public void GetWorldFrame(float t, out float3 position, out quaternion rotation)
    {
        position = EvaluateWorldPosition(t);
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
        var keys = Keys;
        if (keys.Count == 0) return 0f;
        if (keys.Count == 1) return keys[0].facingYawOffset;

        // Keypoints are in ascending parameter order, so the bracket is the last one at or before t.
        var previous = -1;
        for (var i = 0; i < keys.Count; i++)
        {
            if (keys[i].splineT > t) break;
            previous = i;
        }

        if (previous < 0) return closed ? LerpAcross(keys.Count - 1, 0, t) : keys[0].facingYawOffset;
        if (previous == keys.Count - 1)
        {
            return closed ? LerpAcross(keys.Count - 1, 0, t) : keys[^1].facingYawOffset;
        }
        return LerpAcross(previous, previous + 1, t);
    }

    /// <summary>Interpolates the yaw offset between two keypoints, handling the wrap of a closed
    /// spline's final span and the wrap of the angles themselves.</summary>
    private float LerpAcross(int fromIndex, int toIndex, float t)
    {
        var from = Keys[fromIndex];
        var to = Keys[toIndex];

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

    private float3 WorldTangent(float t) => Flatten(EvaluateWorldTangent(t));

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
        var keys = Keys;
        if (!HasPath || keys.Count == 0) return;

        DrawPath();

        for (var i = 0; i < keys.Count; i++)
        {
            var position = EvaluateWorldPosition(keys[i].splineT);

            // The drawn keypoint stands out, so the index field stays findable on a long path even
            // when its skeleton is off-screen.
            Gizmos.color = i == drawKeypointSkeletonAtIndex ? Color.white : Color.yellow;
            Gizmos.DrawSphere(position, 0.05f);

            Gizmos.color = Color.yellow;
            var facing = GetWorldFacing(keys[i].splineT);
            GizmosExtensions.DrawArrow(position, position + facing * 0.4f, 0.1f, thickness: 3);
        }

        DrawKeypointSkeleton();
    }

    /// <summary>
    /// The curve itself, sampled into a polyline. Drawn here because nothing else does any more:
    /// the free Scene-view rendering came from the <c>SplineContainer</c> component this class no
    /// longer keeps.
    /// </summary>
    private void DrawPath()
    {
        const int segments = 256;

        Gizmos.color = new Color(0.3f, 0.3f, 1f);
        var previous = (Vector3)EvaluateWorldPosition(0f);
        for (var i = 1; i <= segments; i++)
        {
            var current = (Vector3)EvaluateWorldPosition(i / (float)segments);
            Gizmos.DrawLine(previous, current);
            previous = current;
        }
    }

    /// <summary>
    /// The pose at <see cref="drawKeypointSkeletonAtIndex"/>, drawn in place on the path as
    /// parent-to-child bone segments — the same shape, and the same red, as the
    /// <c>SkeletonAnimation</c> preview and the other skeleton gizmos in the project.
    /// </summary>
    private void DrawKeypointSkeleton()
    {
        if (drawKeypointSkeletonAtIndex < 0 || drawKeypointSkeletonAtIndex >= Keys.Count) return;

        // Sized from the rig, which answers 0 rather than throwing when it is unconfigured; the pose
        // itself is validated inside TryGetWorldBonePositions.
        var boneCount = Skeleton?.BoneCount ?? 0;
        if (boneCount == 0) return;

        // Temp rather than a cached buffer: one skeleton is solved per repaint, and a MonoBehaviour
        // drawing gizmos in edit mode has no dependable hook to dispose a persistent one from.
        using var world = new NativeArray<float3>(boneCount, Allocator.Temp);
        if (!TryGetWorldBonePositions(Keys[drawKeypointSkeletonAtIndex], world)) return;

        var parentIndices = Skeleton.GetSkeletonData().ParentIndices;
        Gizmos.color = Color.red;
        for (var i = 1; i < boneCount; i++)
        {
            GizmosExtensions.DrawLine(world[parentIndices[i]], world[i], 3);
        }
    }
#endif
}
}
