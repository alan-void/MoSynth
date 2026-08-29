using AnimationTools.Editor;
using NUnit.Framework;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
public class RandomBenchmarkPathTests
{
    private const int SeedSampleCount = 200;

    private static readonly RandomPathKind[] AllKinds =
    {
        RandomPathKind.SmoothLoop,
        RandomPathKind.SharpLoop,
        RandomPathKind.SmoothLine,
        RandomPathKind.SharpLine
    };

    private static RandomPathSettings Settings => RandomPathSettings.Default;

    [Test]
    public void SameSeedAndIndexProduceIdenticalKnots()
    {
        foreach (var kind in AllKinds)
        {
            var first = RandomPathShapes.Generate(kind, 4821, 3, Settings);
            var second = RandomPathShapes.Generate(kind, 4821, 3, Settings);

            Assert.IsNotNull(first, $"{kind} produced no path.");
            Assert.AreEqual(first.Count, second.Count, $"{kind} knot count differed between runs.");

            for (var i = 0; i < first.Count; i++)
            {
                Assert.IsTrue(first[i].Position.Equals(second[i].Position),
                    $"{kind} knot {i} differed between two runs of the same seed.");
            }
        }
    }

    [Test]
    public void DifferentSeedsProduceDifferentPaths()
    {
        var first = RandomPathShapes.Generate(RandomPathKind.SmoothLoop, 4821, 0, Settings);
        var second = RandomPathShapes.Generate(RandomPathKind.SmoothLoop, 4822, 0, Settings);

        Assert.IsFalse(first[0].Position.Equals(second[0].Position),
            "Two different seeds produced the same first knot.");
    }

    [Test]
    public void APathsGeometryDependsOnlyOnItsSeedAndSlot()
    {
        // The property per-slot sub-seeding buys: changing the mix must not disturb the paths that
        // keep their kind. A single shared RNG stream would break this silently.
        var slot = 2;
        var kind = RandomPathShapes.KindForIndex(slot, 8, 1f, 1f);
        Assert.AreEqual(RandomPathKind.SmoothLoop, kind);

        var fromAllSmooth = RandomPathShapes.Generate(kind, 99, slot, Settings);
        var fromHalfSmooth = RandomPathShapes.Generate(
            RandomPathShapes.KindForIndex(slot, 8, 0.5f, 1f), 99, slot, Settings);

        Assert.AreEqual(RandomPathKind.SmoothLoop, RandomPathShapes.KindForIndex(slot, 8, 0.5f, 1f));
        for (var i = 0; i < fromAllSmooth.Count; i++)
        {
            Assert.IsTrue(fromAllSmooth[i].Position.Equals(fromHalfSmooth[i].Position),
                $"Knot {i} changed when the batch ratio changed.");
        }
    }

    [Test]
    public void SmoothPathsStayAboveTheMinimumTurnRadius()
    {
        var settings = Settings;
        for (var seed = 1; seed <= SeedSampleCount; seed++)
        {
            foreach (var kind in new[] { RandomPathKind.SmoothLoop, RandomPathKind.SmoothLine })
            {
                var spline = RandomPathShapes.Generate(kind, seed, 0, settings);
                Assert.IsNotNull(spline, $"{kind} seed {seed} produced no path.");

                var radius = RandomPathShapes.MinTurnRadius(spline, RandomPathShapes.SampleSpacingMeters);
                Assert.GreaterOrEqual(radius, settings.minTurnRadius,
                    $"{kind} seed {seed} turns tighter than the minimum radius.");
            }
        }
    }

    [Test]
    public void NoGeneratedPathSelfIntersects()
    {
        var settings = Settings;
        for (var seed = 1; seed <= SeedSampleCount; seed++)
        {
            foreach (var kind in AllKinds)
            {
                var spline = RandomPathShapes.Generate(kind, seed, 0, settings);
                Assert.IsNotNull(spline, $"{kind} seed {seed} produced no path.");
                Assert.IsFalse(
                    RandomPathShapes.SelfIntersects(spline, RandomPathShapes.SampleSpacingMeters,
                        settings.minSelfClearance, settings.minTurnRadius),
                    $"{kind} seed {seed} crosses or crowds itself.");
            }
        }
    }

    [Test]
    public void SegmentsIntersectDetectsAKnownCrossing()
    {
        // Without this the self-intersection tests would all pass on a predicate stuck at false.
        Assert.IsTrue(RandomPathShapes.SegmentsIntersect(
            new float2(-1f, 0f), new float2(1f, 0f),
            new float2(0f, -1f), new float2(0f, 1f)));
    }

    [Test]
    public void SegmentsIntersectIgnoresSharedAndSeparatedEndpoints()
    {
        Assert.IsFalse(RandomPathShapes.SegmentsIntersect(
            new float2(0f, 0f), new float2(1f, 0f),
            new float2(1f, 0f), new float2(1f, 1f)), "A shared endpoint is not a crossing.");

        Assert.IsFalse(RandomPathShapes.SegmentsIntersect(
            new float2(0f, 0f), new float2(1f, 0f),
            new float2(0f, 1f), new float2(1f, 1f)), "Parallel segments do not cross.");
    }

    [Test]
    public void SharpPathCornersAreNeitherCrowdedNorReversals()
    {
        var settings = Settings;
        for (var seed = 1; seed <= SeedSampleCount; seed++)
        {
            foreach (var kind in new[] { RandomPathKind.SharpLoop, RandomPathKind.SharpLine })
            {
                var spline = RandomPathShapes.Generate(kind, seed, 0, settings);
                Assert.IsNotNull(spline, $"{kind} seed {seed} produced no path.");

                Assert.GreaterOrEqual(RandomPathShapes.MinSegmentLength(spline), settings.minSegmentLength,
                    $"{kind} seed {seed} has a corner with no run-up.");
                Assert.LessOrEqual(RandomPathShapes.MaxTurnAngleDegrees(spline), 140f,
                    $"{kind} seed {seed} has a near-reversal rather than a corner.");
            }
        }
    }

    [Test]
    public void OpenPathsAdvanceMonotonicallyAlongX()
    {
        // The invariant the open families' simplicity rests on.
        for (var seed = 1; seed <= SeedSampleCount; seed++)
        {
            foreach (var kind in new[] { RandomPathKind.SmoothLine, RandomPathKind.SharpLine })
            {
                var spline = RandomPathShapes.Generate(kind, seed, 0, Settings);
                for (var i = 1; i < spline.Count; i++)
                {
                    Assert.Greater(spline[i].Position.x, spline[i - 1].Position.x,
                        $"{kind} seed {seed} steps backwards at knot {i}.");
                }
            }
        }
    }

    [Test]
    public void LoopsAreClosedAndLinesAreNot()
    {
        // BenchmarkLapProbe branches on exactly this to pick lap counting or far-end completion.
        Assert.IsTrue(RandomPathShapes.Generate(RandomPathKind.SmoothLoop, 7, 0, Settings).Closed);
        Assert.IsTrue(RandomPathShapes.Generate(RandomPathKind.SharpLoop, 7, 0, Settings).Closed);
        Assert.IsFalse(RandomPathShapes.Generate(RandomPathKind.SmoothLine, 7, 0, Settings).Closed);
        Assert.IsFalse(RandomPathShapes.Generate(RandomPathKind.SharpLine, 7, 0, Settings).Closed);
    }

    [Test]
    public void EveryPathLiesInThePlane()
    {
        foreach (var kind in AllKinds)
        {
            var spline = RandomPathShapes.Generate(kind, 12345, 0, Settings);
            foreach (var knot in spline) Assert.AreEqual(0f, knot.Position.y, 1e-6f);
        }
    }

    [Test]
    public void KindPartitionCoversEverySlotAndHonoursTheRatios()
    {
        const int count = 8;
        var kinds = new RandomPathKind[count];
        for (var i = 0; i < count; i++) kinds[i] = RandomPathShapes.KindForIndex(i, count, 0.5f, 0.5f);

        Assert.AreEqual(2, System.Array.FindAll(kinds, k => k == RandomPathKind.SmoothLoop).Length);
        Assert.AreEqual(2, System.Array.FindAll(kinds, k => k == RandomPathKind.SmoothLine).Length);
        Assert.AreEqual(2, System.Array.FindAll(kinds, k => k == RandomPathKind.SharpLoop).Length);
        Assert.AreEqual(2, System.Array.FindAll(kinds, k => k == RandomPathKind.SharpLine).Length);
    }

    [Test]
    public void PathNamesCarryTheSeedSlotAndKind()
    {
        Assert.AreEqual("Random_s4821_03_SharpLoop",
            RandomBenchmarkPathGenerator.NameFor(4821, 3, RandomPathKind.SharpLoop));

        // Zero padding is what keeps slot 3 ahead of slot 10 under the driver's ordinal sort.
        Assert.Less(
            string.CompareOrdinal(
                RandomBenchmarkPathGenerator.NameFor(1, 3, RandomPathKind.SmoothLoop),
                RandomBenchmarkPathGenerator.NameFor(1, 10, RandomPathKind.SmoothLoop)),
            0);
    }
}
}
