using NUnit.Framework;
using UnityEngine;

namespace MotionMatching.Tests
{
/// <summary>
/// Guards the one thing that decides whether an open benchmark path can ever finish: the reference
/// point must stop at the end of an open path instead of wrapping back to its start.
/// </summary>
public class SplineControlInputFoldTests
{
    private const float Tolerance = 1e-6f;

    /// <summary>Exposes the protected seam and lets a test state the closedness directly, with no container to build.</summary>
    private class FoldProbe : SplineControlInput
    {
        public bool closed;

        protected override bool IsClosed => closed;

        public float FoldAt(float t) => Fold(t);
    }

    private GameObject _host;
    private FoldProbe _probe;

    [SetUp]
    public void SetUp()
    {
        _host = new GameObject(nameof(SplineControlInputFoldTests));
        _host.SetActive(false);
        _probe = _host.AddComponent<FoldProbe>();
    }

    [TearDown]
    public void TearDown()
    {
        Object.DestroyImmediate(_host);
    }

    [Test]
    public void ClosedPathWrapsPastTheSeam()
    {
        _probe.closed = true;
        Assert.AreEqual(0.25f, _probe.FoldAt(1.25f), Tolerance);
        Assert.AreEqual(0.9f, _probe.FoldAt(-0.1f), Tolerance);
    }

    [Test]
    public void OpenPathClampsAtTheFarEnd()
    {
        _probe.closed = false;
        Assert.AreEqual(1f, _probe.FoldAt(1.25f), Tolerance);
        Assert.AreEqual(0f, _probe.FoldAt(-0.1f), Tolerance);
    }

    [Test]
    public void BothLeaveAParameterInsideThePathAlone()
    {
        foreach (var closed in new[] { true, false })
        {
            _probe.closed = closed;
            Assert.AreEqual(0.4f, _probe.FoldAt(0.4f), Tolerance, $"closed: {closed}");
        }
    }
}
}
