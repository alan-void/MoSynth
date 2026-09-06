using System;
using AnimationTools;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching.Tests
{
/// <summary>
/// Every control input answers the query in the character's own frame, through one pair of helpers
/// on the base class. A sign or an axis wrong here is a query that asks for the opposite of what the
/// player wants, on every input at once, so the conversion is pinned rather than eyeballed.
/// </summary>
public class ControlInputPlanarHelpersTests
{
    private const float Tolerance = 1e-4f;

    /// <summary>
    /// Reaches the helpers, which are protected because they are plumbing for subclasses rather
    /// than public API. Nothing is instantiated: they are static.
    /// </summary>
    private class Probe : MotionMatchingControlInput
    {
        public static void Position(Transform character, float2 world, Span<float> output) =>
            WritePlanarPosition(character, world, output);

        public static void Direction(Transform character, float2 world, Span<float> output) =>
            WritePlanarDirection(character, world, output);

        public static float2 Planar(Transform t) => PlanarPosition(t);
        public static float2 Forward(Transform t) => PlanarForward(t);

        protected override void OnUpdate() { }
        public override float3 GetWorldInitPosition() => float3.zero;
        public override float3 GetWorldInitDirection() => math.forward();
        public override float3 GetPosition() => float3.zero;
        public override float GetTargetSpeed() => 0f;

        public override void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index,
            Transform character, Span<float> span)
        {
        }
    }

    private GameObject _characterObject;
    private Transform _character;

    [SetUp]
    public void SetUp()
    {
        _characterObject = new GameObject("Character");
        _character = _characterObject.transform;
        _character.SetPositionAndRotation(new Vector3(2f, 0f, 3f), Quaternion.Euler(0f, 90f, 0f));
    }

    [TearDown]
    public void TearDown()
    {
        UnityEngine.Object.DestroyImmediate(_characterObject);
    }

    /// <summary>
    /// A point one metre along world +x, from a character standing at (2, 3) facing +x, is one
    /// metre behind it: the offset is (-1, 0) in world, and the character's own forward is +x.
    /// </summary>
    [Test]
    public void WritePlanarPosition_MeasuresTheOffsetInTheCharactersFrame()
    {
        Span<float> output = stackalloc float[2];
        Probe.Position(_character, new float2(1f, 3f), output);

        Assert.That(output[0], Is.EqualTo(0f).Within(Tolerance), "no sideways offset");
        Assert.That(output[1], Is.EqualTo(-1f).Within(Tolerance), "one metre behind");
    }

    [Test]
    public void WritePlanarPosition_OfTheCharactersOwnPosition_IsTheOrigin()
    {
        Span<float> output = stackalloc float[2];
        Probe.Position(_character, new float2(2f, 3f), output);

        Assert.That(output[0], Is.EqualTo(0f).Within(Tolerance));
        Assert.That(output[1], Is.EqualTo(0f).Within(Tolerance));
    }

    /// <summary>A direction carries no position, so the character's own facing reads as forward.</summary>
    [Test]
    public void WritePlanarDirection_OfTheCharactersFacing_IsForward()
    {
        Span<float> output = stackalloc float[2];
        Probe.Direction(_character, new float2(1f, 0f), output);

        Assert.That(output[0], Is.EqualTo(0f).Within(Tolerance));
        Assert.That(output[1], Is.EqualTo(1f).Within(Tolerance));
    }

    [Test]
    public void WritePlanarDirection_OfWorldForward_IsToTheCharactersLeft()
    {
        Span<float> output = stackalloc float[2];
        Probe.Direction(_character, new float2(0f, 1f), output);

        Assert.That(output[0], Is.EqualTo(-1f).Within(Tolerance));
        Assert.That(output[1], Is.EqualTo(0f).Within(Tolerance));
    }

    [Test]
    public void PlanarPositionAndForward_DropTheVerticalAxis()
    {
        _character.position = new Vector3(2f, 5f, 3f);
        _character.rotation = Quaternion.Euler(30f, 0f, 0f);

        var position = Probe.Planar(_character);
        Assert.That(position.x, Is.EqualTo(2f).Within(Tolerance));
        Assert.That(position.y, Is.EqualTo(3f).Within(Tolerance));

        var forward = Probe.Forward(_character);
        Assert.That(math.length(forward), Is.EqualTo(1f).Within(Tolerance), "flattened but still unit");
        Assert.That(forward.y, Is.EqualTo(1f).Within(Tolerance), "a pitched character still faces +z");
    }

    /// <summary>A character aimed straight up has no facing to flatten, so a default stands in.</summary>
    [Test]
    public void PlanarForward_OfAVerticalFacing_FallsBackToWorldForward()
    {
        _character.rotation = Quaternion.Euler(-90f, 0f, 0f);

        var forward = Probe.Forward(_character);
        Assert.That(math.length(forward), Is.EqualTo(1f).Within(Tolerance));
    }
}
}
