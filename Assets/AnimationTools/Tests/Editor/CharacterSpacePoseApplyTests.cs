using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
/// <summary>
/// Writing a character-frame pose back into a <see cref="PoseBuffer"/>.
/// </summary>
/// <remarks>
/// The property that matters is that <see cref="CharacterSpacePose.Apply"/> undoes
/// <see cref="CharacterSpacePose.Extract"/> exactly. A stage running a learned model measures the
/// pose one way and writes the prediction back the other, so any gap between the two is a bias the
/// model can never learn its way out of — and, like every disagreement in this area, it would not
/// throw.
/// </remarks>
public class CharacterSpacePoseApplyTests
{
    private const float DeltaTime = 1f / 30f;
    private const float RootHeight = 0.9f;

    private Skeleton _skeleton;
    private SkeletonData _skeletonData;
    private SimulationFrameDef _frameDef;
    private PoseBuffer _pose;
    private PoseBuffer _rebuilt;

    private NativeArray<float3> _positions;
    private NativeArray<quaternion> _rotations;
    private NativeArray<float3> _velocities;
    private NativeArray<float3> _angularVelocities;

    [SetUp]
    public void SetUp()
    {
        _skeleton = TestSkeletons.CreateChain3();
        _skeletonData = _skeleton.GetSkeletonData();
        _frameDef = SimulationFrameDef.Default(_skeleton);

        var layout = PoseLayoutBuilder.Build(_skeleton, out _);
        _pose = PoseBuffer.Allocate(layout, Allocator.Temp);
        _rebuilt = PoseBuffer.Allocate(layout, Allocator.Temp);

        var positions = _pose.Positions;
        var rotations = _pose.Rotations;
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            positions[i] = i == 0
                ? new float3(0f, RootHeight, 0f)
                : _skeleton.GetBone(i).RestLocalPosition;
            rotations[i] = quaternion.identity;
        }

        var boneCount = _skeleton.BoneCount;
        _positions = new NativeArray<float3>(boneCount, Allocator.Temp);
        _rotations = new NativeArray<quaternion>(boneCount, Allocator.Temp);
        _velocities = new NativeArray<float3>(boneCount, Allocator.Temp);
        _angularVelocities = new NativeArray<float3>(boneCount, Allocator.Temp);
    }

    [TearDown]
    public void TearDown()
    {
        _positions.Dispose();
        _rotations.Dispose();
        _velocities.Dispose();
        _angularVelocities.Dispose();
        _rebuilt.Dispose();
        _pose.Dispose();
        TestSkeletons.DestroyAll();
    }

    /// <summary>A pose that is not symmetric in any axis, so a swapped sign cannot hide.</summary>
    private void SetAwkwardPose()
    {
        var positions = _pose.Positions;
        var rotations = _pose.Rotations;
        var velocities = _pose.Velocities;
        var angularVelocities = _pose.AngularVelocities;

        positions[0] = new float3(3.5f, RootHeight, -1.25f);
        rotations[0] = quaternion.Euler(0.11f, 0.7f, -0.05f);
        rotations[1] = quaternion.Euler(-0.2f, 0.13f, 0.31f);
        rotations[2] = quaternion.Euler(0.4f, -0.25f, 0.08f);

        velocities[0] = new float3(0.4f, -0.1f, 1.1f);
        angularVelocities[0] = new float3(0.05f, 0.9f, -0.2f);
        velocities[1] = new float3(0.02f, 0.03f, -0.01f);
        angularVelocities[1] = new float3(-0.3f, 0.15f, 0.22f);
        velocities[2] = new float3(-0.05f, 0.01f, 0.04f);
        angularVelocities[2] = new float3(0.12f, -0.4f, 0.07f);
    }

    /// <summary>Extract this pose, then write it straight back into <see cref="_rebuilt"/>.</summary>
    private void RoundTrip(bool withRates)
    {
        CharacterSpacePose.Extract(_pose, _skeletonData, _frameDef, DeltaTime,
            _positions, _rotations,
            withRates ? _velocities : default, withRates ? _angularVelocities : default);

        SimulationFrame.Compute(_pose, _skeletonData, _frameDef,
            out var framePosition, out var frameRotation);
        SimulationFrame.ComputeVelocity(_pose, _skeletonData, _frameDef, DeltaTime,
            out var frameLinearVelocity, out var frameYawRate);

        CharacterSpacePose.Apply(_rebuilt, _skeletonData, framePosition, frameRotation,
            frameLinearVelocity, frameYawRate, _positions, _rotations,
            withRates ? _velocities : default, withRates ? _angularVelocities : default);
    }

    private static void AssertSameRotation(quaternion expected, quaternion actual, string what)
    {
        // A quaternion double-covers SO(3), so compare the rotation rather than the four floats.
        var dot = math.abs(math.dot(expected.value, actual.value));
        Assert.That(dot, Is.EqualTo(1f).Within(1e-4f), what);
    }

    [Test]
    public void ApplyUndoesExtract()
    {
        SetAwkwardPose();

        RoundTrip(withRates: true);

        var original = _pose.Positions;
        var rebuilt = _rebuilt.Positions;
        var originalRotations = _pose.Rotations;
        var rebuiltRotations = _rebuilt.Rotations;

        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            Assert.That(math.distance(original[i], rebuilt[i]), Is.LessThan(1e-4f),
                $"bone {i} position");
            AssertSameRotation(originalRotations[i], rebuiltRotations[i], $"bone {i} rotation");
        }
    }

    [Test]
    public void ApplyUndoesExtractForTheRateChannelsToo()
    {
        SetAwkwardPose();

        RoundTrip(withRates: true);

        var originalVelocities = _pose.Velocities;
        var rebuiltVelocities = _rebuilt.Velocities;
        var originalAngular = _pose.AngularVelocities;
        var rebuiltAngular = _rebuilt.AngularVelocities;

        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            Assert.That(math.distance(originalVelocities[i], rebuiltVelocities[i]),
                Is.LessThan(1e-3f), $"bone {i} linear velocity");
            Assert.That(math.distance(originalAngular[i], rebuiltAngular[i]),
                Is.LessThan(1e-3f), $"bone {i} angular velocity");
        }
    }

    [Test]
    public void RatesAreOptionalOnTheWayBack()
    {
        SetAwkwardPose();

        RoundTrip(withRates: false);

        var original = _pose.Positions;
        var rebuilt = _rebuilt.Positions;
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            Assert.That(math.distance(original[i], rebuilt[i]), Is.LessThan(1e-4f), $"bone {i}");
        }
    }

    [Test]
    public void BonesBelowTheRootKeepTheirRestOffsets()
    {
        // Only rotations are written for them, which is what stops a predicted pose stretching a
        // bone. Feeding in positions that disagree must not move the joint.
        SetAwkwardPose();
        CharacterSpacePose.Extract(_pose, _skeletonData, _frameDef, DeltaTime, _positions, _rotations);

        _positions[1] += new float3(0.5f, 0.5f, 0.5f);
        _positions[2] += new float3(-0.3f, 0.2f, 0.9f);

        CharacterSpacePose.Apply(_rebuilt, _skeletonData, float3.zero, quaternion.identity,
            float3.zero, 0f, _positions, _rotations);

        var rebuilt = _rebuilt.Positions;
        for (var i = 1; i < _skeleton.BoneCount; i++)
        {
            Assert.That(math.distance(rebuilt[i], _skeleton.GetBone(i).RestLocalPosition),
                Is.LessThan(1e-5f), $"bone {i}");
        }
    }

    [Test]
    public void TheFrameVelocityComesBackOutOfBoneZero()
    {
        // How a stage makes the character travel: it writes the frame's own motion into bone 0's
        // velocity channels, and MotionSynthesisComponent reads it straight back with ComputeVelocity.
        SetAwkwardPose();
        CharacterSpacePose.Extract(_pose, _skeletonData, _frameDef, DeltaTime,
            _positions, _rotations, _velocities, _angularVelocities);

        var frameVelocity = new float3(0.2f, 0f, 1.4f);
        const float yawRate = 0.6f;
        SimulationFrame.Compute(_pose, _skeletonData, _frameDef,
            out var framePosition, out var frameRotation);

        CharacterSpacePose.Apply(_rebuilt, _skeletonData, framePosition, frameRotation,
            frameVelocity, yawRate, _positions, _rotations, _velocities, _angularVelocities);

        SimulationFrame.ComputeVelocity(_rebuilt, _skeletonData, _frameDef, DeltaTime,
            out var readBack, out var readBackYawRate);

        Assert.That(math.distance(readBack, frameVelocity), Is.LessThan(1e-3f), "frame velocity");
        Assert.That(readBackYawRate, Is.EqualTo(yawRate).Within(1e-3f), "frame yaw rate");
    }
}
}
