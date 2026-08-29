using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine.Splines;
using Random = Unity.Mathematics.Random;

namespace AnimationTools.Editor
{
/// <summary>The four random path families. Closedness and tangent mode both follow from the kind.</summary>
public enum RandomPathKind
{
    SmoothLoop,
    SharpLoop,
    SmoothLine,
    SharpLine
}

/// <summary>Bounds a generated path has to satisfy to be worth benchmarking on.</summary>
public struct RandomPathSettings
{
    /// <summary>Smallest and largest base radius (loops) or step spacing scale (lines), in meters.</summary>
    public float extentMin;
    public float extentMax;

    /// <summary>Tightest turn a smooth path may demand, in meters. Below this no locomotion clip can follow it.</summary>
    public float minTurnRadius;

    /// <summary>Shortest straight between two corners of a sharp path, in meters.</summary>
    public float minSegmentLength;

    /// <summary>How close two distant parts of the same path may come, in meters.</summary>
    public float minSelfClearance;

    public static RandomPathSettings Default => new()
    {
        extentMin = 5f,
        extentMax = 9f,
        minTurnRadius = 1.5f,
        minSegmentLength = 2f,
        minSelfClearance = 1f
    };
}

/// <summary>
/// Generates the random benchmark path families, and the geometric predicates that decide whether a
/// candidate is followable. Pure: nothing here touches the AssetDatabase or a scene.
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

    private const int MaxAttempts = 32;
    private const int MaxRelaxations = 3;
    private const float RelaxationFactor = 0.8f;

    private const float SharpLoopMaxTurnDegrees = 140f;
    private const float SharpLineMaxTurnDegrees = 120f;

    /// <summary>
    /// The path for one slot of a batch, or null if no candidate met the settings. Drawing from a
    /// stream seeded on <paramref name="seed"/> and <paramref name="index"/> together — rather than
    /// one stream shared across the batch — is what keeps a path's geometry independent of how many
    /// other paths were generated alongside it, or of what kind they were.
    /// </summary>
    public static Spline Generate(RandomPathKind kind, int seed, int index, in RandomPathSettings settings)
    {
        var rng = Random.CreateFromIndex(math.hash(new uint2((uint)seed, (uint)index)));

        // Retrying from the same stream keeps the attempt count itself deterministic.
        var wobbleScale = 1f;
        for (var relaxation = 0; relaxation <= MaxRelaxations; relaxation++)
        {
            for (var attempt = 0; attempt < MaxAttempts; attempt++)
            {
                var spline = Build(kind, ref rng, settings, wobbleScale);
                if (IsAcceptable(spline, kind, settings)) return spline;
            }

            wobbleScale *= RelaxationFactor;
        }

        return null;
    }

    /// <summary>Which family the slot at <paramref name="index"/> belongs to.</summary>
    /// <remarks>
    /// A partition rather than a draw, so a kind is a pure function of the ratios and the slot. It
    /// also groups the folder: all the smooth loops, then the smooth lines, and so on.
    /// </remarks>
    public static RandomPathKind KindForIndex(int index, int count, float smoothRatio, float closedRatio)
    {
        var smoothCount = (int)math.round(count * math.saturate(smoothRatio));
        var isSmooth = index < smoothCount;

        var familyCount = isSmooth ? smoothCount : count - smoothCount;
        var familyIndex = isSmooth ? index : index - smoothCount;
        var closedCount = (int)math.round(familyCount * math.saturate(closedRatio));
        var isClosed = familyIndex < closedCount;

        if (isSmooth) return isClosed ? RandomPathKind.SmoothLoop : RandomPathKind.SmoothLine;
        return isClosed ? RandomPathKind.SharpLoop : RandomPathKind.SharpLine;
    }

    public static bool IsClosed(RandomPathKind kind) =>
        kind is RandomPathKind.SmoothLoop or RandomPathKind.SharpLoop;

    private static Spline Build(RandomPathKind kind, ref Random rng, in RandomPathSettings settings, float wobbleScale)
    {
        return kind switch
        {
            RandomPathKind.SmoothLoop => SmoothLoop(ref rng, settings, wobbleScale),
            RandomPathKind.SharpLoop => SharpLoop(ref rng, settings, wobbleScale),
            RandomPathKind.SmoothLine => SmoothLine(ref rng, settings, wobbleScale),
            _ => SharpLine(ref rng, settings, wobbleScale)
        };
    }

    private static bool IsAcceptable(Spline spline, RandomPathKind kind, in RandomPathSettings settings)
    {
        if (SelfIntersects(spline, SampleSpacingMeters, settings.minSelfClearance, settings.minTurnRadius)) return false;

        if (kind is RandomPathKind.SmoothLoop or RandomPathKind.SmoothLine)
            return MinTurnRadius(spline, SampleSpacingMeters) >= settings.minTurnRadius;

        // A corner is a curvature singularity by construction, so the demand on the sharp families is
        // instead that corners stay far enough apart to be separable and never become a reversal.
        if (MinSegmentLength(spline) < settings.minSegmentLength) return false;

        var cap = kind == RandomPathKind.SharpLoop ? SharpLoopMaxTurnDegrees : SharpLineMaxTurnDegrees;
        return MaxTurnAngleDegrees(spline) <= cap;
    }

    // ---- Closed families -------------------------------------------------------------------

    /// <summary>
    /// Knots at strictly increasing angles around the origin with a perturbed radius, so the knot
    /// polygon is star-shaped and therefore simple.
    /// </summary>
    /// <remarks>
    /// The wobble amplitude is derived from the minimum turn radius rather than chosen as a fraction
    /// of the radius: modelling the wobble as a sinusoid of wavelength 2*pi*R/n gives a peak curvature
    /// of roughly 1/R + a*n^2/R^2, so the n^2 makes a "percentage of radius" amplitude unfollowable at
    /// even moderate knot counts. This only sizes the draw — the curvature test is the guarantee.
    /// </remarks>
    private static Spline SmoothLoop(ref Random rng, in RandomPathSettings settings, float wobbleScale)
    {
        var n = rng.NextInt(5, 9);
        var radius = rng.NextFloat(settings.extentMin, settings.extentMax);

        var maxAmplitude = radius * (radius - settings.minTurnRadius) / (settings.minTurnRadius * n * n);
        var amplitude = rng.NextFloat(0.35f * maxAmplitude, 0.90f * maxAmplitude) * wobbleScale;

        var points = new float3[n];
        for (var i = 0; i < n; i++)
        {
            var angle = 2f * math.PI * i / n + rng.NextFloat(-1f, 1f) * (0.6f * math.PI / n);
            var r = radius + amplitude * rng.NextFloat(-1f, 1f);
            points[i] = new float3(r * math.cos(angle), 0f, r * math.sin(angle));
        }

        return FromPoints(points, TangentMode.AutoSmooth, closed: true);
    }

    /// <summary>The same star-shaped construction with a wider radial swing and linear tangents, so the spline is exactly its knot polygon.</summary>
    private static Spline SharpLoop(ref Random rng, in RandomPathSettings settings, float wobbleScale)
    {
        var n = rng.NextInt(4, 9);
        var radius = rng.NextFloat(settings.extentMin, settings.extentMax);
        var swing = rng.NextFloat(0.25f, 0.55f) * wobbleScale;

        var points = new float3[n];
        for (var i = 0; i < n; i++)
        {
            var angle = 2f * math.PI * i / n + rng.NextFloat(-1f, 1f) * (0.6f * math.PI / n);
            var r = radius * (1f + swing * rng.NextFloat(-1f, 1f));
            points[i] = new float3(r * math.cos(angle), 0f, r * math.sin(angle));
        }

        return FromPoints(points, TangentMode.Linear, closed: true);
    }

    // ---- Open families ---------------------------------------------------------------------

    /// <summary>
    /// A corridor advancing along +X with a lateral wobble.
    /// </summary>
    /// <remarks>
    /// Deliberately not an unclosed ring: that would leave the start and end one segment apart, so
    /// "reached the far end" would be indistinguishable from "back at the start" and a pure-pursuit
    /// lookahead near the end would point across the gap. Because X strictly increases, the polyline
    /// is the graph of a function of X and so cannot cross itself.
    /// </remarks>
    private static Spline SmoothLine(ref Random rng, in RandomPathSettings settings, float wobbleScale)
    {
        var n = rng.NextInt(6, 11);
        var step = rng.NextFloat(3.5f, 5.5f);

        // Peak curvature of a lateral sinusoid of wavelength 2*step is about pi^2*A/step^2.
        var maxAmplitude = step * step / (math.PI * math.PI * settings.minTurnRadius);
        var amplitude = rng.NextFloat(0.35f, 0.90f) * maxAmplitude * wobbleScale;

        return FromPoints(Corridor(ref rng, n, step, amplitude), TangentMode.AutoSmooth, closed: false);
    }

    /// <summary>The same corridor with linear tangents; the amplitude cap keeps each corner under about 120 degrees.</summary>
    private static Spline SharpLine(ref Random rng, in RandomPathSettings settings, float wobbleScale)
    {
        var n = rng.NextInt(5, 10);
        var step = rng.NextFloat(3.5f, 6f);
        var amplitude = rng.NextFloat(0.40f, 1f) * (0.87f * step) * wobbleScale;

        return FromPoints(Corridor(ref rng, n, step, amplitude), TangentMode.Linear, closed: false);
    }

    /// <summary>Jitter under half the step keeps X strictly increasing, which is what rules out self-intersection.</summary>
    private static float3[] Corridor(ref Random rng, int n, float step, float amplitude)
    {
        var points = new float3[n];
        for (var i = 0; i < n; i++)
        {
            var x = i * step + rng.NextFloat(-1f, 1f) * 0.25f * step;
            var z = rng.NextFloat(-amplitude, amplitude);
            points[i] = new float3(x, 0f, z);
        }

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
    /// sampled at <paramref name="sampleSpacingMeters"/>. Meaningless on a linear-tangent spline,
    /// where a corner is a genuine curvature singularity.
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
    /// slide onto the wrong one, which a lap counter then reads as a seam crossing. Pairs nearer than
    /// a minimum-radius U-turn along the path are exempt, since those are legitimately close in space.
    /// </remarks>
    public static bool SelfIntersects(Spline spline, float sampleSpacingMeters, float minClearanceMeters, float minTurnRadiusMeters)
    {
        var samples = Sample(spline, sampleSpacingMeters);
        var count = samples.Count;
        if (count < 4) return false;

        var closed = spline.Closed;
        var segments = closed ? count : count - 1;
        var spacing = spline.GetLength() / segments;
        var exemptArcLength = math.PI * minTurnRadiusMeters;

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
