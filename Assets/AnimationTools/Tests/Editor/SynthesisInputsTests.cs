using NUnit.Framework;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
public class SynthesisInputsTests
{
    [Test]
    public void AFullBufferDropsMarkersInsteadOfGrowing()
    {
        var inputs = new SynthesisInputs("test", 2, 1);
        inputs.AddPoint(SynthesisInputRole.Request, new float3(1f, 0f, 0f));
        inputs.AddArrow(SynthesisInputRole.Answer, float3.zero, new float3(0f, 0f, 2f));
        inputs.AddPoint(SynthesisInputRole.State, new float3(9f, 9f, 9f));

        Assert.AreEqual(2, inputs.MarkerCount);
        Assert.IsFalse(inputs.GetMarker(0).IsArrow);
        Assert.IsTrue(inputs.GetMarker(1).IsArrow);
        Assert.AreEqual(SynthesisInputRole.Answer, inputs.GetMarker(1).Role);
        Assert.AreEqual(2f, inputs.GetMarker(1).Vector.z);
    }

    [Test]
    public void BeginTickClearsOnlyTheEventFlags()
    {
        var inputs = new SynthesisInputs("test", 4, 4);
        inputs.AddPoint(SynthesisInputRole.Request, float3.zero);
        inputs.AddReadout("cost", 1.5f, "0.0");
        inputs.Searched = true;
        inputs.AnswerChanged = true;

        inputs.BeginTick();

        Assert.IsFalse(inputs.Searched);
        Assert.IsFalse(inputs.AnswerChanged);
        Assert.AreEqual(1, inputs.MarkerCount, "a matcher's query stays drawn between searches");
        Assert.AreEqual(1, inputs.ReadoutCount);
    }
}
}
