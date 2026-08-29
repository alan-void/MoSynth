using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// Measuring a pose in its own character frame.
/// </summary>
/// <remarks>
/// The fixtures are rigid translations and rotations, because those are the cases where the right
/// answer is known exactly: a character carried along by its own frame has no motion *within* that
/// frame, whatever the world says it is doing. They deliberately mirror
/// <c>Python/tests/test_training_data.py</c> — the two implementations have to agree on this
/// definition, and a mismatch in frame, units or rate convention does not throw.
/// </remarks>
public class CharacterSpacePoseTests
{
    private const float DeltaTime = 1f / 30f;
    private const float RootHeight = 0.9f;

    private Skeleton _skeleton;
    private SkeletonData _skeletonData;
    private SimulationFrameDef _frameDef;
    private PoseBuffer _pose;

    private NativeArray<float3> _positions;
    private NativeArray<quaternion> _rotations;
    private NativeArray<float3> _velocities;
    private NativeArray<float3> _angularVelocities;

    /// <summary>root -&gt; spine -&gt; head, a metre apart, so a bone offset is easy to read.</summary>
    [SetUp]
    public void SetUp()
    {
        _skeleton = TestSkeletons.CreateChain3();
        _skeletonData = _skeleton.GetSkeletonData();
        _frameDef = SimulationFrameDef.Default(_skeleton);

        _pose = PoseBuffer.Allocate(PoseLayoutBuilder.Build(_skeleton, out _), Allocator.Temp);
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
        _pose.Dispose();
        TestSkeletons.DestroyAll();
    }

    private void SetPosition(int bone, float3 value)
    {
        var positions = _pose.Positions;
        positions[bone] = value;
    }

    private void SetRotation(int bone, quaternion value)
    {
        var rotations = _pose.Rotations;
        rotations[bone] = value;
    }

    private void SetVelocity(int bone, float3 value)
    {
        var velocities = _pose.Velocities;
        velocities[bone] = value;
    }

    private void SetAngularVelocity(int bone, float3 value)
    {
        var angularVelocities = _pose.AngularVelocities;
        angularVelocities[bone] = value;
    }

    private void Extract() => CharacterSpacePose.Extract(_pose, _skeletonData, _frameDef, DeltaTime,
        _positions, _rotations, _velocities, _angularVelocities);

    /// <summary>The whole rig translating rigidly at <paramref name="velocity"/> metres per second.</summary>
    private void SetRigidTranslation(float3 velocity)
    {
        var velocities = _pose.Velocities;
        var angularVelocities = _pose.AngularVelocities;
        velocities[0] = velocity;
        for (var i = 1; i < _skeleton.BoneCount; i++)
        {
            velocities[i] = float3.zero;
            angularVelocities[i] = float3.zero;
        }
    }

    [Test]
    public void TheRootBoneSitsOverTheFrameOrigin()
    {
        // The frame is bone 0 ground-projected, so bone 0 can only be directly above it. If this
        // drifts, the frame transform is wrong.
        SetPosition(0, new float3(7f, RootHeight, -3f));

        Extract();

        Assert.That(math.abs(_positions[0].x), Is.LessThan(1e-5f));
        Assert.That(math.abs(_positions[0].z), Is.LessThan(1e-5f));
        Assert.That(_positions[0].y, Is.EqualTo(RootHeight).Within(1e-5f));
    }

    [Test]
    public void BonesKeepTheirRestOffsets()
    {
        Extract();

        // spine is one metre above the root, head one above that.
        Assert.That(_positions[1].y, Is.EqualTo(RootHeight + 1f).Within(1e-5f));
        Assert.That(_positions[2].y, Is.EqualTo(RootHeight + 2f).Within(1e-5f));
    }

    [Test]
    public void FacingIsRemovedFromEveryBone()
    {
        // Turn the whole rig a quarter turn. The bones sit on the frame's own axis, so measuring
        // them inside that frame has to give the same answer as before the turn.
        var yaw = quaternion.RotateY(0.5f * math.PI);
        SetRotation(0, yaw);
        SetPosition(0, new float3(2f, RootHeight, 5f));

        Extract();

        Assert.That(math.length(_positions[2] - new float3(0f, RootHeight + 2f, 0f)), Is.LessThan(1e-4f));
        Assert.That(math.abs(math.dot(_rotations[2], quaternion.identity)), Is.EqualTo(1f).Within(1e-4f));
    }

    [Test]
    public void ARigidTranslationLeavesNoMotionInsideTheFrame()
    {
        SetRigidTranslation(new float3(1.5f, 0f, 0f));

        Extract();

        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            Assert.That(math.length(_velocities[i]), Is.LessThan(1e-4f), $"bone {i} linear");
            Assert.That(math.length(_angularVelocities[i]), Is.LessThan(1e-4f), $"bone {i} angular");
        }
    }

    [Test]
    public void ARigidTurnLeavesNoMotionInsideTheFrame()
    {
        var yawRate = 0.5f;
        SetAngularVelocity(0, new float3(0f, yawRate, 0f));
        for (var i = 1; i < _skeleton.BoneCount; i++)
        {
            SetVelocity(i, float3.zero);
            SetAngularVelocity(i, float3.zero);
        }

        // Bone 0 sits on the frame's own axis, so a turn about that axis moves it nowhere.
        SetVelocity(0, float3.zero);

        Extract();

        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            Assert.That(math.length(_velocities[i]), Is.LessThan(1e-3f), $"bone {i} linear");
            Assert.That(math.length(_angularVelocities[i]), Is.LessThan(1e-3f), $"bone {i} angular");
        }
    }

    [Test]
    public void MotionOfOneBoneRelativeToTheFrameSurvives()
    {
        // The spine alone rotates; the root is still. That is real motion inside the frame, and
        // the head above it must pick up the tangential velocity.
        SetAngularVelocity(1, new float3(0f, 0f, 1f));

        Extract();

        Assert.That(math.length(_angularVelocities[0]), Is.LessThan(1e-4f));
        Assert.That(math.length(_angularVelocities[1] - new float3(0f, 0f, 1f)), Is.LessThan(1e-4f));
        // The head is one metre above the spine, turning about +Z at 1 rad/s.
        Assert.That(math.length(_velocities[2]), Is.EqualTo(1f).Within(1e-3f));
    }

    [Test]
    public void RatesCanBeSkipped()
    {
        SetRigidTranslation(new float3(1.5f, 0f, 0f));

        CharacterSpacePose.Extract(_pose, _skeletonData, _frameDef, DeltaTime, _positions, _rotations);

        Assert.That(_positions[1].y, Is.EqualTo(RootHeight + 1f).Within(1e-5f));
    }

    [Test]
    public void PositionsAndRotationsAgreeWithTheSingleBoneQueries()
    {
        // The forward pass has to give the same answers as walking each bone's chain separately.
        SetRotation(0, quaternion.Euler(0f, 0.7f, 0f));
        SetRotation(1, quaternion.Euler(0.3f, 0f, 0.2f));
        SetPosition(0, new float3(1f, RootHeight, 2f));

        Extract();

        SimulationFrame.Compute(_pose, _skeletonData, _frameDef, out var framePosition, out var frameRotation);
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            var expectedPosition = SimulationFrame.ToFrameLocal(
                _skeletonData.CharacterSpacePosition(_pose, i), framePosition, frameRotation);
            var expectedRotation = SimulationFrame.ToFrameLocal(
                _skeletonData.CharacterSpaceRotation(_pose, i), frameRotation);

            Assert.That(math.distance(_positions[i], expectedPosition), Is.LessThan(1e-4f), $"bone {i} position");
            Assert.That(math.abs(math.dot(_rotations[i], expectedRotation)), Is.EqualTo(1f).Within(1e-4f),
                $"bone {i} rotation");
        }
    }

    [Test]
    public void WorldRatesAgreeWithTheSingleBoneQueries()
    {
        SetRotation(1, quaternion.Euler(0.3f, 0f, 0.2f));
        SetVelocity(0, new float3(0.4f, 0f, -0.2f));
        SetAngularVelocity(0, new float3(0f, 0.6f, 0f));
        SetAngularVelocity(1, new float3(0.1f, 0f, 0.5f));
        SetVelocity(2, new float3(0f, 0.05f, 0f));

        Extract();

        SimulationFrame.Compute(_pose, _skeletonData, _frameDef, out var framePosition, out var frameRotation);
        SimulationFrame.ComputeVelocity(_pose, _skeletonData, _frameDef, DeltaTime,
            out var frameLinearVelocity, out var frameYawRate);

        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            SimulationFrame.DecomposeRootVelocity(framePosition, frameRotation, frameLinearVelocity,
                frameYawRate, _skeletonData.CharacterSpacePosition(_pose, i),
                _skeletonData.CharacterSpaceVelocity(_pose, i),
                _skeletonData.CharacterSpaceAngularVelocity(_pose, i),
                out var expectedLinear, out var expectedAngular);

            Assert.That(math.distance(_velocities[i], expectedLinear), Is.LessThan(1e-4f), $"bone {i} linear");
            Assert.That(math.distance(_angularVelocities[i], expectedAngular), Is.LessThan(1e-4f),
                $"bone {i} angular");
        }
    }
}
}
