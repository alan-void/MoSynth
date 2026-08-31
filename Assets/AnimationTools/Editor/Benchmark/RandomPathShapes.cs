using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine.Splines;
using Random = Unity.Mathematics.Random;

namespace AnimationTools.Editor
{
/// <summary>How a random path is laid out in the plane, independent of whether its turns are rounded or cornered.</summary>
public enum RandomPathShape
{
    /// <summary>A closed ring about the origin.</summary>
    Loop,

    /// <summary>An open corridor advancing along +X.</summary>
    Line,

    /// <summary>An open path that turns freely and is bounded by nothing.</summary>
    Walk
}

/// <summary>The six random path families: each shape with rounded turns, and again with corners.</summary>
public enum RandomPathKind
{
    SmoothLoop,
    SmoothLine,
    SmoothWalk,
    SharpLoop,
    SharpLine,
    SharpWalk
}

/// <summary>Size and separation bounds a generated path is drawn within.</summary>
public struct RandomPathSettings
{
    /// <summary>Smallest and largest overall path size, in meters: a ring's radius, and the length an open path is scaled to match.</summary>
    public float extentMin;
    public float extentMax;

    /// <summary>Fewest and most control points a path may be built from.</summary>
    public int knotCountMin;
    public int knotCountMax;

    /// <summary>Shortest straight between two corners of a sharp path, in meters.</summary>
    public float minSegmentLength;

    /// <summary>How close two distant parts of the same path may come, in meters.</summary>
    public float minSelfClearance;

    public static RandomPathSettings Default => new()
    {
        extentMin = 5f,
        extentMax = 9f,
        knotCountMin = 5,
        knotCountMax = 12,
        minSegmentLength = 2f,
        minSelfClearance = 1f
    };
}

/// <summary>
/// Generates the random benchmark path families, and the geometric predicates that decide whether a
/// candidate is usable. Pure: nothing here touches the AssetDatabase or a scene.
/// </summary>
/// <remarks>
/// Geometry is a pure function of <c>(kind, seed, index)</c>, which is what lets a prefab name carry
/// its own reproduction recipe. See <c>openwiki/animation-tools/benchmarking.md</c> for why the
/// rejection tests exist and what each one protects.
/// </remarks>
public static class RandomPathShapes
{
    /// <summary>Polyline resolution for every sampled predicate, in meters.</summary>
    public const float SampleSpacingMeters = 0.25f;

    /// <summary>Fewest knots any shape can be built from; a two-knot loop is a degenerate there-and-back.</summary>
    public const int MinKnotCount = 3;

    private const int MaxAttempts = 32;
    private const int MaxRelaxations = 3;
    private const float RelaxationFactor = 0.8f;

    private const float SharpLoopMaxTurnDegrees = 140f;
    private const float SharpOpenMaxTurnDegrees = 120f;

    // Radial swing of a ring, as a fraction of its radius.
    private const float SmoothRingSwingMin = 0.06f;
    private const float SmoothRingSwingMax = 0.22f;
    private const float SharpRingSwingMin = 0.25f;
    private const float SharpRingSwingMax = 0.55f;

    // Lateral swing of a corridor, as a fraction of its step.
    private const float SmoothCorridorSwingMin = 0.15f;
    private const float SmoothCorridorSwingMax = 0.45f;
    private const float SharpCorridorSwingMin = 0.35f;
    private const float SharpCorridorSwingMax = 0.87f;

    // Largest heading change a walk may make at one knot.
    private const float SmoothWalkTurnDegrees = 45f;
    private const float SharpWalkTurnDegrees = 100f;

    /// <summary>
    /// The path for one slot of a batch, or null if no candidate met the settings. Drawing from a
    /// stream seeded on <paramref name="seed"/> and <paramref name="index"/> together -- rather than
    /// one stream shared across the batch -- is what keeps a path's geometry independent of how many
    /// other paths were generated alongside it, or of what kind they were.
    /// </summary>
    public static Spline Generate(RandomPathKind kind, int seed, int index, in RandomPathSettings settings) =>
        Generate(kind, seed, index, settings, out _);

    /// <summary>
    /// As <see cref="Generate(RandomPathKind,int,int,in RandomPathSettings)"/>, also reporting why the
    /// last candidate was turned down. An unsatisfiable combination of settings is easy to ask for --
    /// twenty knots inside a five meter extent leaves the sharp families no room between corners -- so
    /// naming the predicate that failed is what tells the caller which setting to move.
    /// </summary>
    public static Spline Generate(RandomPathKind kind, int seed, int index, in RandomPathSettings settings,
        out string rejection)
    {
        var rng = Random.CreateFromIndex(math.hash(new uint2((uint)seed, (uint)index)));

        // Retrying from the same stream keeps the attempt count itself deterministic.
        rejection = "no candidate was built";
        var wobbleScale = 1f;
        for (var relaxation = 0; relaxation <= MaxRelaxations; relaxation++)
        {
            for (var attempt = 0; attempt < MaxAttempts; attempt++)
            {
                var spline = Build(kind, ref rng, settings, wobbleScale);
                rejection = Rejection(spline, kind, settings);
                if (rejection == null) return spline;
            }

            wobbleScale *= RelaxationFactor;
        }

        return null;
    }

    /// <summary>Which family the slot at <paramref name="index"/> belongs to.</summary>
    /// <remarks>
    /// A partition rather than a draw, so a kind is a pure function of the ratios and the slot. The
    /// two levels nest: the batch splits into smooth and sharp, and each of those into loops, then
    /// corridors, then walks -- so <paramref name="walkRatio"/> is a share of what does not loop, and
    /// each ratio stays independently meaningful. Emitting in that order also groups the folder.
    /// </remarks>
    public static RandomPathKind KindForIndex(int index, int count, float smoothRatio, float closedRatio,
        float walkRatio)
    {
        var smoothCount = (int)math.round(count * math.saturate(smoothRatio));
        var isSmooth = index < smoothCount;

        var blockCount = isSmooth ? smoothCount : count - smoothCount;
        var blockIndex = isSmooth ? index : index - smoothCount;

        var loopCount = (int)math.round(blockCount * math.saturate(closedRatio));
        var openCount = blockCount - loopCount;
        var walkCount = (int)math.round(openCount * math.saturate(walkRatio));
        var lineCount = openCount - walkCount;

        var shape = blockIndex < loopCount ? RandomPathShape.Loop
            : blockIndex < loopCount + lineCount ? RandomPathShape.Line
            : RandomPathShape.Walk;

        return KindFor(isSmooth, shape);
    }

    public static RandomPathKind KindFor(bool smooth, RandomPathShape shape) => shape switch
    {
        RandomPathShape.Loop => smooth ? RandomPathKind.SmoothLoop : RandomPathKind.SharpLoop,
        RandomPathShape.Line => smooth ? RandomPathKind.SmoothLine : RandomPathKind.SharpLine,
        _ => smooth ? RandomPathKind.SmoothWalk : RandomPathKind.SharpWalk
    };

    public static RandomPathShape ShapeOf(RandomPathKind kind) => kind switch
    {
        RandomPathKind.SmoothLoop or RandomPathKind.SharpLoop => RandomPathShape.Loop,
        RandomPathKind.SmoothLine or RandomPathKind.SharpLine => RandomPathShape.Line,
        _ => RandomPathShape.Walk
    };

    /// <summary>Whether the family is auto-smoothed. A sharp family's spline is exactly its knot polygon.</summary>
    public static bool IsSmooth(RandomPathKind kind) =>
        kind is RandomPathKind.SmoothLoop or RandomPathKind.SmoothLine or RandomPathKind.SmoothWalk;

    public static bool IsClosed(RandomPathKind kind) => ShapeOf(kind) == RandomPathShape.Loop;

    private static Spline Build(RandomPathKind kind, ref Random rng, in RandomPathSettings settings, float wobbleScale)
    {
        var smooth = IsSmooth(kind);
        var shape = ShapeOf(kind);
        var knotCount = rng.NextInt(KnotCountMin(settings), KnotCountMax(settings) + 1);
        var extent = rng.NextFloat(settings.extentMin, settings.extentMax);

        var points = shape switch
        {
            RandomPathShape.Loop => Ring(ref rng, knotCount, extent, smooth, wobbleScale),
            RandomPathShape.Line => Corridor(ref rng, knotCount, StepFor(extent, knotCount), smooth, wobbleScale),
            _ => Walk(ref rng, knotCount, StepFor(extent, knotCount), smooth, wobbleScale)
        };

        return FromPoints(points, smooth ? TangentMode.AutoSmooth : TangentMode.Linear,
            closed: shape == RandomPathShape.Loop);
    }

    /// <summary>Why the candidate is unusable, or null if it is fine.</summary>
    private static string Rejection(Spline spline, RandomPathKind kind, in RandomPathSettings settings)
    {
        if (SelfIntersects(spline, SampleSpacingMeters, settings.minSelfClearance))
            return $"it crosses or comes within {settings.minSelfClearance:0.00} m of itself";

        // Nothing bounds how tightly a smooth path may turn. The sharp families never had such a bound
        // either, and how well a method follows a demanding curve is the measurement, not a defect --
        // so the generator reports the tightest turn it drew instead of rejecting it.
        if (IsSmooth(kind)) return null;

        // A corner is a curvature singularity by construction, so the demand on the sharp families is
        // instead that corners stay far enough apart to be separable and never become a reversal.
        var separation = MinSegmentLength(spline);
        if (separation < settings.minSegmentLength)
        {
            return $"corners {separation:0.00} m apart are below the {settings.minSegmentLength:0.00} m minimum";
        }

        var cap = ShapeOf(kind) == RandomPathShape.Loop ? SharpLoopMaxTurnDegrees : SharpOpenMaxTurnDegrees;
        var turn = MaxTurnAngleDegrees(spline);
        return turn > cap ? $"a {turn:0} deg turn exceeds the {cap:0} deg maximum" : null;
    }

    private static int KnotCountMin(in RandomPathSettings settings) =>
        math.max(MinKnotCount, settings.knotCountMin);

    private static int KnotCountMax(in RandomPathSettings settings) =>
        math.max(KnotCountMin(settings), settings.knotCountMax);

    /// <summary>
    /// Step length giving an open path the same total length as a ring of the same extent, so the
    /// extent means one thing -- how big the path is -- across all six families, and a sweep's runs
    /// stay comparable in duration.
    /// </summary>
    private static float StepFor(float extent, int knotCount) => 2f * math.PI * extent / knotCount;

    // ---- Shapes ----------------------------------------------------------------------------

    /// <summary>
    /// Knots at strictly increasing angles around the origin with a perturbed radius, so the knot
    /// polygon is star-shaped and therefore simple.
    /// </summary>
    private static float3[] Ring(ref Random rng, int knotCount, float radius, bool smooth, float wobbleScale)
    {
        var swing = wobbleScale * (smooth
            ? rng.NextFloat(SmoothRingSwingMin, SmoothRingSwingMax)
            : rng.NextFloat(SharpRingSwingMin, SharpRingSwingMax));

        var points = new float3[knotCount];
        for (var i = 0; i < knotCount; i++)
        {
            // Angular jitter below the nominal gap leaves the angles strictly increasing.
            var angle = 2f * math.PI * i / knotCount + rng.NextFloat(-1f, 1f) * (0.6f * math.PI / knotCount);
            var r = radius * (1f + swing * rng.NextFloat(-1f, 1f));
            points[i] = new float3(r * math.cos(angle), 0f, r * math.sin(angle));
        }

        return points;
    }

    /// <summary>
    /// A corridor advancing along +X with a lateral wobble.
    /// </summary>
    /// <remarks>
    /// Deliberately not an unclosed ring: that would leave the start and end one segment apart, so
    /// "reached the far end" would be indistinguishable from "back at the start" and a pure-pursuit
    /// lookahead near the end would point across the gap. Because X strictly increases, the polyline
    /// is the graph of a function of X and so cannot cross itself.
    /// </remarks>
    private static float3[] Corridor(ref Random rng, int knotCount, float step, bool smooth, float wobbleScale)
    {
        var amplitude = step * wobbleScale * (smooth
            ? rng.NextFloat(SmoothCorridorSwingMin, SmoothCorridorSwingMax)
            : rng.NextFloat(SharpCorridorSwingMin, SharpCorridorSwingMax));

        var points = new float3[knotCount];
        for (var i = 0; i < knotCount; i++)
        {
            // Jitter under half the step is what keeps X strictly increasing.
            var x = i * step + rng.NextFloat(-1f, 1f) * 0.25f * step;
            var z = rng.NextFloat(-amplitude, amplitude);
            points[i] = new float3(x, 0f, z);
        }

        return points;
    }

    /// <summary>
    /// A path that turns by a random amount at every knot, contained by nothing, so it neither orbits
    /// nor marches along an axis.
    /// </summary>
    /// <remarks>
    /// Unlike the ring and the corridor this has no construction-level simplicity argument -- a
    /// wanderer really can cross itself -- so it leans entirely on the rejection loop. Relaxation
    /// shrinks the heading cap, which straightens the walk, and is what makes the retries converge.
    /// </remarks>
    private static float3[] Walk(ref Random rng, int knotCount, float step, bool smooth, float wobbleScale)
    {
        var turnCap = math.radians(smooth ? SmoothWalkTurnDegrees : SharpWalkTurnDegrees) * wobbleScale;
        var heading = rng.NextFloat(0f, 2f * math.PI);

        var points = new float3[knotCount];
        for (var i = 1; i < knotCount; i++)
        {
            heading += rng.NextFloat(-turnCap, turnCap);
            var length = step * rng.NextFloat(0.8f, 1.2f);
            points[i] = points[i - 1] + new float3(length * math.cos(heading), 0f, length * math.sin(heading));
        }

        return Recentre(points);
    }

    /// <summary>Shifts a drifting path so its bounding box is centred on the origin, where the other shapes already sit.</summary>
    private static float3[] Recentre(float3[] points)
    {
        var min = points[0];
        var max = points[0];
        foreach (var point in points)
        {
            min = math.min(min, point);
            max = math.max(max, point);
        }

        var centre = 0.5f * (min + max);
        for (var i = 0; i < points.Length; i++) points[i] -= centre;
        return points;
    }

    private static Spline FromPoints(IReadOnlyList<float3> points, TangentMode mode, bool closed)
    {
        var spline = new Spline { Closed = closed };
        foreach (var point in points) spline.Add(new BezierKnot(point), mode);
        return spline;
    }

    // ---- Predicates ------------------------------------------------------------------------

    /// <summary>
    /// Tightest turn anywhere on the path, in meters, as the smallest circumradius over a polyline
    /// sampled at <paramref name="sampleSpacingMeters"/>. Diagnostic rather than enforced: nothing
    /// rejects a candidate for turning tightly, and on a linear-tangent spline a corner is a genuine
    /// curvature singularity, so the figure only means anything for the smooth families.
    /// </summary>
    public static float MinTurnRadius(Spline spline, float sampleSpacingMeters)
    {
        var samples = Sample(spline, sampleSpacingMeters);
        if (samples.Count < 3) return float.MaxValue;

        var closed = spline.Closed;
        var last = closed ? samples.Count : samples.Count - 2;

        var minimum = float.MaxValue;
        for (var i = 0; i < last; i++)
        {
            var a = samples[i];
            var b = samples[(i + 1) % samples.Count];
            var c = samples[(i + 2) % samples.Count];
            minimum = math.min(minimum, Circumradius(a, b, c));
        }

        return minimum;
    }

    /// <summary>Shortest distance between consecutive knots, in meters.</summary>
    public static float MinSegmentLength(Spline spline)
    {
        if (spline.Count < 2) return float.MaxValue;

        var segments = spline.Closed ? spline.Count : spline.Count - 1;
        var minimum = float.MaxValue;
        for (var i = 0; i < segments; i++)
        {
            var a = Flatten(spline[i].Position);
            var b = Flatten(spline[(i + 1) % spline.Count].Position);
            minimum = math.min(minimum, math.distance(a, b));
        }

        return minimum;
    }

    /// <summary>Sharpest heading change at any knot, in degrees. Zero on a path with no corners.</summary>
    public static float MaxTurnAngleDegrees(Spline spline)
    {
        if (spline.Count < 3) return 0f;

        var closed = spline.Closed;
        var count = spline.Count;
        var maximum = 0f;

        for (var i = 0; i < count; i++)
        {
            if (!closed && (i == 0 || i == count - 1)) continue;

            var previous = Flatten(spline[(i - 1 + count) % count].Position);
            var current = Flatten(spline[i].Position);
            var next = Flatten(spline[(i + 1) % count].Position);

            var incoming = math.normalizesafe(current - previous);
            var outgoing = math.normalizesafe(next - current);
            if (math.lengthsq(incoming) < 0.5f || math.lengthsq(outgoing) < 0.5f) continue;

            var turn = math.degrees(math.acos(math.clamp(math.dot(incoming, outgoing), -1f, 1f)));
            maximum = math.max(maximum, turn);
        }

        return maximum;
    }

    /// <summary>
    /// Whether the path crosses itself, or approaches itself closer than
    /// <paramref name="minClearanceMeters"/> at points far apart along the path.
    /// </summary>
    /// <remarks>
    /// The clearance half is the one that matters in practice: <see cref="SplineProjector"/> already
    /// survives a clean crossing, but two branches running close and parallel let its windowed search
    /// slide onto the wrong one, which a lap counter then reads as a seam crossing. Pairs near each
    /// other along the path are exempt, being legitimately close in space: the tightest U-turn whose
    /// arms are exactly the clearance apart has half that as its radius, so its arc is a little over
    /// pi times the clearance.
    /// </remarks>
    public static bool SelfIntersects(Spline spline, float sampleSpacingMeters, float minClearanceMeters)
    {
        var samples = Sample(spline, sampleSpacingMeters);
        var count = samples.Count;
        if (count < 4) return false;

        var closed = spline.Closed;
        var segments = closed ? count : count - 1;
        var spacing = spline.GetLength() / segments;
        var exemptArcLength = math.PI * minClearanceMeters;

        for (var i = 0; i < segments; i++)
        {
            var a0 = samples[i];
            var a1 = samples[(i + 1) % count];

            for (var j = i + 2; j < segments; j++)
            {
                // On a closed path the last segment is adjacent to the first.
                if (closed && i == 0 && j == segments - 1) continue;

                var b0 = samples[j];
                var b1 = samples[(j + 1) % count];
                if (SegmentsIntersect(a0, a1, b0, b1)) return true;

                var separation = ArcSeparation(i, j, count, closed) * spacing;
                if (separation >= exemptArcLength && math.distance(a0, b0) < minClearanceMeters) return true;
            }
        }

        return false;
    }

    /// <summary>Whether two XZ segments properly cross. Shared endpoints and touching endpoints do not count.</summary>
    public static bool SegmentsIntersect(float2 a0, float2 a1, float2 b0, float2 b1)
    {
        var d1 = Cross(b1 - b0, a0 - b0);
        var d2 = Cross(b1 - b0, a1 - b0);
        var d3 = Cross(a1 - a0, b0 - a0);
        var d4 = Cross(a1 - a0, b1 - a0);

        return ((d1 > 0f && d2 < 0f) || (d1 < 0f && d2 > 0f)) &&
               ((d3 > 0f && d4 < 0f) || (d3 < 0f && d4 > 0f));
    }

    private static int ArcSeparation(int i, int j, int count, bool closed)
    {
        var forward = j - i;
        return closed ? math.min(forward, count - forward) : forward;
    }

    private static float Cross(float2 a, float2 b) => a.x * b.y - a.y * b.x;

    private static float2 Flatten(float3 point) => new(point.x, point.z);

    /// <summary>
    /// The path as an XZ polyline. Normalized spline parameters are uniform in arc length, so
    /// stepping t uniformly gives evenly spaced samples on any spline.
    /// </summary>
    private static List<float2> Sample(Spline spline, float sampleSpacingMeters)
    {
        var length = spline.GetLength();
        var segments = math.max(8, (int)math.ceil(length / math.max(1e-3f, sampleSpacingMeters)));

        // A closed path's last sample would land back on the first, so it is left out and the
        // wrap-around segment closes the loop instead.
        var count = spline.Closed ? segments : segments + 1;
        var samples = new List<float2>(count);
        for (var i = 0; i < count; i++)
        {
            var t = (float)i / segments;
            samples.Add(Flatten(spline.EvaluatePosition(t)));
        }

        return samples;
    }

    private static float Circumradius(float2 a, float2 b, float2 c)
    {
        var ab = math.distance(a, b);
        var bc = math.distance(b, c);
        var ca = math.distance(c, a);

        var area = 0.5f * math.abs(Cross(b - a, c - a));
        if (area < 1e-9f) return float.MaxValue;

        return ab * bc * ca / (4f * area);
    }
}
}
