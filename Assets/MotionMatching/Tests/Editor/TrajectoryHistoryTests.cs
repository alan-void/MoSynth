using NUnit.Framework;
using Unity.Mathematics;

namespace MotionMatching.Tests
{
public class TrajectoryHistoryTests
{
    private static readonly float2 Forward = new(0f, 1f);

    /// <summary>Records <paramref name="count"/> samples marching along +X at 1 m/s from t = 0.</summary>
    private static TrajectoryHistory Marching(int count, int capacity, float step = 0.1f)
    {
        var history = new TrajectoryHistory(capacity);
        for (var i = 0; i < count; i++)
        {
            history.Record(i * step, new float2(i * step, 0f), Forward);
        }

        return history;
    }

    [Test]
    public void TrySample_BeforeAnythingIsRecorded_Fails()
    {
        Assert.IsFalse(new TrajectoryHistory(8).TrySample(0f, out _, out _));
    }

    [Test]
    public void TrySample_BetweenTwoSamples_Interpolates()
    {
        var history = Marching(count: 5, capacity: 8);

        Assert.IsTrue(history.TrySample(0.25f, out var position, out _));
        Assert.That(position.x, Is.EqualTo(0.25f).Within(1e-5f));
    }

    [Test]
    public void TrySample_AtOrAfterTheNewest_ReportsTheNewest()
    {
        var history = Marching(count: 5, capacity: 8);

        Assert.IsTrue(history.TrySample(99f, out var position, out _));
        Assert.That(position.x, Is.EqualTo(0.4f).Within(1e-5f));
    }

    [Test]
    public void TrySample_OlderThanTheBufferHolds_Fails()
    {
        // Ten samples through a four-slot ring: everything before t = 0.6 has been dropped.
        var history = Marching(count: 10, capacity: 4);

        Assert.IsFalse(history.TrySample(0.3f, out _, out _),
            "Clamping here would report the start of the run as a long stand still.");
        Assert.IsTrue(history.TrySample(0.7f, out var position, out _));
        Assert.That(position.x, Is.EqualTo(0.7f).Within(1e-5f));
    }

    [Test]
    public void Record_WithANonAdvancingTimestamp_ReplacesTheNewestSample()
    {
        var history = new TrajectoryHistory(8);
        history.Record(0f, float2.zero, Forward);
        history.Record(1f, new float2(1f, 0f), Forward);
        history.Record(1f, new float2(5f, 0f), Forward);

        Assert.AreEqual(2, history.Count);
        Assert.IsTrue(history.TrySample(1f, out var position, out _));
        Assert.That(position.x, Is.EqualTo(5f).Within(1e-5f));
    }

    [Test]
    public void TrySample_InterpolatesFacingAsAUnitVector()
    {
        var history = new TrajectoryHistory(8);
        history.Record(0f, float2.zero, new float2(1f, 0f));
        history.Record(1f, float2.zero, new float2(0f, 1f));

        Assert.IsTrue(history.TrySample(0.5f, out _, out var forward));
        Assert.That(math.length(forward), Is.EqualTo(1f).Within(1e-5f));
        Assert.That(forward.x, Is.EqualTo(forward.y).Within(1e-5f));
    }

    [Test]
    public void Clear_ForgetsEverything()
    {
        var history = Marching(count: 5, capacity: 8);
        history.Clear();

        Assert.AreEqual(0, history.Count);
        Assert.IsFalse(history.TrySample(0.2f, out _, out _));
    }

    [Test]
    public void Capacity_IsAtLeastTwoSoAQueryCanAlwaysBeBracketed()
    {
        Assert.AreEqual(2, new TrajectoryHistory(0).Capacity);
    }
}
}
