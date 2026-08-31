using System;
using AnimationTools.Editor;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Splines;

namespace AnimationTools.Tests
{
public class RandomBenchmarkPathTests
{
    private const int SeedSampleCount = 200;

    private static readonly RandomPathKind[] AllKinds =
    {
        RandomPathKind.SmoothLoop,
        RandomPathKind.SmoothLine,
        RandomPathKind.SmoothWalk,
        RandomPathKind.SharpLoop,
        RandomPathKind.SharpLine,
        RandomPathKind.SharpWalk
    };

    private static readonly RandomPathKind[] SharpKinds =
    {
        RandomPathKind.SharpLoop,
        RandomPathKind.SharpLine,
        RandomPathKind.SharpWalk
    };

    private static readonly RandomPathKind[] WalkKinds =
    {
        RandomPathKind.SmoothWalk,
        RandomPathKind.SharpWalk
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
        const int slot = 2;
        var kind = RandomPathShapes.KindForIndex(slot, 8, 1f, 1f, 0f);
        Assert.AreEqual(RandomPathKind.SmoothLoop, kind);
        Assert.AreEqual(RandomPathKind.SmoothLoop, RandomPathShapes.KindForIndex(slot, 8, 0.5f, 1f, 0.5f));

        var fromAllSmooth = RandomPathShapes.Generate(kind, 99, slot, Settings);
        var fromHalfSmooth = RandomPathShapes.Generate(
            RandomPathShapes.KindForIndex(slot, 8, 0.5f, 1f, 0.5f), 99, slot, Settings);

        for (var i = 0; i < fromAllSmooth.Count; i++)
        {
            Assert.IsTrue(fromAllSmooth[i].Position.Equals(fromHalfSmooth[i].Position),
                $"Knot {i} changed when the batch ratios changed.");
        }
    }

    [Test]
    public void EverySeedProducesAPath()
    {
        // The rejection loop is the only thing standing between a settings combination and an empty
        // slot, and the walks lean on it hardest -- they have no construction that rules out crossing.
        foreach (var kind in AllKinds)
        {
            for (var seed = 1; seed <= SeedSampleCount; seed++)
            {
                Assert.IsNotNull(RandomPathShapes.Generate(kind, seed, 0, Settings, out var rejection),
                    $"{kind} seed {seed} exhausted its attempts; last rejection: {rejection}.");
            }
        }
    }

    [Test]
    public void KnotCountsStayInsideTheConfiguredRange()
    {
        foreach (var range in new[] { new int2(3, 4), new int2(5, 12), new int2(9, 14) })
        {
            var settings = Settings;
            settings.knotCountMin = range.x;
            settings.knotCountMax = range.y;

            foreach (var kind in AllKinds)
            {
                for (var seed = 1; seed <= 50; seed++)
                {
                    var spline = RandomPathShapes.Generate(kind, seed, 0, settings);
                    Assert.IsNotNull(spline, $"{kind} seed {seed} produced no path for range {range}.");
                    Assert.GreaterOrEqual(spline.Count, range.x, $"{kind} seed {seed} fell below the knot range.");
                    Assert.LessOrEqual(spline.Count, range.y, $"{kind} seed {seed} exceeded the knot range.");
                }
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
                        settings.minSelfClearance),
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
            foreach (var kind in SharpKinds)
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
    public void CorridorsAdvanceMonotonicallyAlongX()
    {
        // The invariant the corridor families' simplicity rests on. Walks are deliberately excluded:
        // wandering off any single axis is the whole point of them.
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
    public void WalksAreNeitherCorridorsNorLoops()
    {
        // The reason the family exists: nothing contains it. A corridor always sets off along +X and
        // never doubles back on it; a loop ends where it started. Aggregates rather than per-seed
        // assertions, because any one short walk can legitimately resemble either.
        foreach (var kind in WalkKinds)
        {
            var sectors = new bool[8];
            var travelling = 0;
            var doublingBack = 0;

            for (var seed = 1; seed <= SeedSampleCount; seed++)
            {
                var spline = RandomPathShapes.Generate(kind, seed, 0, Settings);
                Assert.IsFalse(spline.Closed, $"{kind} seed {seed} is closed.");

                var start = spline[0].Position;
                var end = spline[spline.Count - 1].Position;
                var displacement = new float2(end.x - start.x, end.z - start.z);

                if (math.length(displacement) > 0.1f * spline.GetLength())
                {
                    travelling++;
                    sectors[Sector(displacement)] = true;
                }

                if (!IsMonotone(spline, position => position.x)) doublingBack++;
            }

            Assert.IsTrue(Array.TrueForAll(sectors, hit => hit),
                $"{kind}s never set off in some direction, so something still contains them.");
            Assert.Greater(travelling, SeedSampleCount / 2,
                $"Most {kind}s end near where they started, which makes them loops.");
            Assert.Greater(doublingBack, SeedSampleCount / 20,
                $"No {kind} ever doubles back along X, which is corridor behaviour.");
        }
    }

    [Test]
    public void LoopsAreClosedAndEverythingElseIsOpen()
    {
        // BenchmarkLapProbe branches on exactly this to pick lap counting or far-end completion.
        foreach (var kind in AllKinds)
        {
            var spline = RandomPathShapes.Generate(kind, 7, 0, Settings);
            Assert.AreEqual(RandomPathShapes.IsClosed(kind), spline.Closed, $"{kind} closedness disagrees with its shape.");
            Assert.AreEqual(RandomPathShapes.ShapeOf(kind) == RandomPathShape.Loop, spline.Closed);
        }
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
    public void KindPartitionHonoursTheThreeRatios()
    {
        const int count = 12;
        var kinds = new RandomPathKind[count];
        for (var i = 0; i < count; i++) kinds[i] = RandomPathShapes.KindForIndex(i, count, 0.5f, 0.34f, 0.5f);

        foreach (var kind in AllKinds)
        {
            Assert.AreEqual(2, Array.FindAll(kinds, k => k == kind).Length,
                $"An even three-way split over 12 slots should give two {kind}s.");
        }
    }

    [Test]
    public void KindPartitionCoversEverySlotWhateverTheRatios()
    {
        foreach (var count in new[] { 1, 5, 12, 99 })
        {
            foreach (var smooth in new[] { 0f, 0.25f, 0.5f, 1f })
            foreach (var closed in new[] { 0f, 0.34f, 1f })
            foreach (var walk in new[] { 0f, 0.5f, 1f })
            {
                var total = 0;
                foreach (var kind in AllKinds)
                {
                    for (var i = 0; i < count; i++)
                    {
                        if (RandomPathShapes.KindForIndex(i, count, smooth, closed, walk) == kind) total++;
                    }
                }

                Assert.AreEqual(count, total,
                    $"Slots went unassigned at count {count}, ratios {smooth}/{closed}/{walk}.");
            }
        }
    }

    [Test]
    public void PathNamesCarryTheSeedSlotAndKind()
    {
        Assert.AreEqual("Random_s4821_03_SharpWalk",
            RandomBenchmarkPathGenerator.NameFor(4821, 3, RandomPathKind.SharpWalk));

        // Zero padding is what keeps slot 3 ahead of slot 10 under the driver's ordinal sort.
        Assert.Less(
            string.CompareOrdinal(
                RandomBenchmarkPathGenerator.NameFor(1, 3, RandomPathKind.SmoothLoop),
                RandomBenchmarkPathGenerator.NameFor(1, 10, RandomPathKind.SmoothLoop)),
            0);
    }

    /// <summary>Which of eight compass sectors a direction falls in.</summary>
    private static int Sector(float2 direction)
    {
        var turns = math.frac(math.atan2(direction.y, direction.x) / (2f * math.PI) + 1f);
        return math.min((int)math.floor(8f * turns), 7);
    }

    private static bool IsMonotone(Spline spline, Func<float3, float> axis)
    {
        var rising = true;
        var falling = true;
        for (var i = 1; i < spline.Count; i++)
        {
            if (axis(spline[i].Position) <= axis(spline[i - 1].Position)) rising = false;
            if (axis(spline[i].Position) >= axis(spline[i - 1].Position)) falling = false;
        }

        return rising || falling;
    }
}
}
