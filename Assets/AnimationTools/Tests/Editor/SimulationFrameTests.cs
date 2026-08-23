using AnimationTools;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// The character frame is derived from a pose rather than stored, so these tests pin the
/// derivation itself: the ground projection and yaw extraction, the finite-step velocity form the
/// extraction pipeline relies on, and the frame-local round trips the runtime re-anchors poses
/// with.
/// </summary>
public class SimulationFrameTests
{
    private const float Tolerance = 1e-4f;
    private const float FrameTime = 1f / 60f;

    private Skeleton _skeleton;
    private PoseLayout _layout;
    private SkeletonData _skeletonData;
    private PoseBuffer _pose;

    [SetUp]
    public void SetUp()
    {
        _skeleton = TestSkeletons.CreateChain3();
        _layout = PoseLayout.CreateFullPose(_skeleton, true, true);
        _skeletonData = _skeleton.GetSkeletonData();
        _pose = AllocateRestPose();
    }

    [TearDown]
    public void TearDown()
    {
        _pose.Dispose();
        TestSkeletons.DestroyAll();
    }

    /// <summary>Rest offsets, identity rotations, zero rates — tests override only what they need.</summary>
    private PoseBuffer AllocateRestPose()
    {
        var pose = PoseBuffer.Allocate(_layout, Allocator.Temp);
        var positions = pose.Positions;
        var rotations = pose.Rotations;

        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            positions[i] = _skeleton.GetBone(i).RestLocalPosition;
            rotations[i] = quaternion.identity;
        }

        return pose;
    }

    private static void SetRoot(PoseBuffer pose, float3 position, quaternion rotation)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;
        positions[0] = position;
        rotations[0] = rotation;
    }

    /// <summary>Finite differences of every channel of <paramref name="pose"/> toward
    /// <paramref name="next"/>, exactly as pose extraction stores them.</summary>
    private static void FillFiniteDifferences(PoseBuffer pose, PoseBuffer next, float dt)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;
        var velocities = pose.Velocities;
        var angularVelocities = pose.AngularVelocities;
        var nextPositions = next.Positions;
        var nextRotations = next.Rotations;

        for (var i = 0; i < positions.Length; i++)
        {
            velocities[i] = (nextPositions[i] - positions[i]) / dt;
            angularVelocities[i] = MathExtensions.AngularVelocity(rotations[i], nextRotations[i], dt);
        }
    }

    private static float YawOf(quaternion rotation)
    {
        var forward = math.mul(rotation, math.forward());
        return math.atan2(forward.x, forward.z);
    }

    private static float WrapPi(float angle)
    {
        while (angle > math.PI) angle -= 2f * math.PI;
        while (angle < -math.PI) angle += 2f * math.PI;
        return angle;
    }

    private static void AssertApprox(float3 expected, float3 actual, string message = null)
    {
        Assert.AreEqual(expected.x, actual.x, Tolerance, message);
        Assert.AreEqual(expected.y, actual.y, Tolerance, message);
        Assert.AreEqual(expected.z, actual.z, Tolerance, message);
    }

    private static void AssertApprox(quaternion expected, quaternion actual, string message = null)
    {
        var expectedValue = expected.value;
        var actualValue = actual.value;
        if (math.dot(expectedValue, actualValue) < 0f) actualValue = -actualValue;

        Assert.AreEqual(expectedValue.x, actualValue.x, Tolerance, message);
        Assert.AreEqual(expectedValue.y, actualValue.y, Tolerance, message);
        Assert.AreEqual(expectedValue.z, actualValue.z, Tolerance, message);
        Assert.AreEqual(expectedValue.w, actualValue.w, Tolerance, message);
    }

    [Test]
    public void Compute_RootWithPitchAndYaw_ProjectsPositionAndKeepsOnlyYaw()
    {
        var rootPosition = new float3(3f, 1.25f, -2f);
        var rootRotation = math.mul(quaternion.RotateY(math.radians(40f)), quaternion.RotateX(math.radians(25f)));
        SetRoot(_pose, rootPosition, rootRotation);

        var def = SimulationFrameDef.Default(_skeleton);
        SimulationFrame.Compute(_pose, _skeletonData, def, out var framePos, out var frameRot);

        AssertApprox(new float3(3f, 0f, -2f), framePos);

        // The reference implementation, spelled with Unity's own quaternion type: the reference
        // bone's forward axis flattened onto the ground plane and aimed at.
        var forward = (Quaternion)rootRotation * (Vector3)def.ForwardAxisLocal;
        forward.y = 0f;
        AssertApprox((quaternion)Quaternion.LookRotation(forward.normalized, Vector3.up), frameRot);

        Assert.AreEqual(0f, frameRot.value.x, Tolerance, "frame rotation must be yaw-only");
        Assert.AreEqual(0f, frameRot.value.z, Tolerance, "frame rotation must be yaw-only");
    }

    [Test]
    public void Compute_NonRootReferenceBone_UsesThatBonesProjectedPosition()
    {
        var rotations = _pose.Rotations;
        SetRoot(_pose, new float3(1f, 0.9f, 0f), quaternion.RotateY(math.radians(15f)));
        rotations[1] = quaternion.RotateZ(math.radians(30f));

        var def = new SimulationFrameDef
        {
            ReferenceBoneIndex = 2,
            ForwardAxisLocal = _skeleton.RestLocalAxis(2, math.forward())
        };

        SimulationFrame.Compute(_pose, _skeletonData, def, out var framePos, out var frameRot);

        var headPosition = _skeletonData.CharacterSpacePosition(_pose, 2);
        AssertApprox(new float3(headPosition.x, 0f, headPosition.z), framePos);

        var headForward = math.mul(_skeletonData.CharacterSpaceRotation(_pose, 2), def.ForwardAxisLocal);
        headForward.y = 0f;
        AssertApprox(quaternion.LookRotationSafe(math.normalize(headForward), math.up()), frameRot);
    }

    [Test]
    public void ComputeVelocity_FiniteDifferencePose_MatchesTheFrameItStepsTo()
    {
        var rootPosition = new float3(2f, 1f, 5f);
        var rootRotation = quaternion.RotateY(math.radians(20f));
        var nextRootPosition = new float3(2.15f, 1.02f, 5.3f);
        var nextRootRotation = math.mul(quaternion.RotateY(math.radians(27f)), quaternion.RotateX(math.radians(4f)));

        SetRoot(_pose, rootPosition, rootRotation);

        var nextPose = AllocateRestPose();
        try
        {
            SetRoot(nextPose, nextRootPosition, nextRootRotation);
            FillFiniteDifferences(_pose, nextPose, FrameTime);

            var def = SimulationFrameDef.Default(_skeleton);
            SimulationFrame.Compute(_pose, _skeletonData, def, out var framePos, out var frameRot);
            SimulationFrame.Compute(nextPose, _skeletonData, def, out var nextFramePos, out var nextFrameRot);

            SimulationFrame.ComputeVelocity(_pose, _skeletonData, def, FrameTime,
                out var linVelFrameLocal, out var yawRate);

            var expectedLinVel = math.mul(math.inverse(frameRot), (nextFramePos - framePos) / FrameTime);
            AssertApprox(expectedLinVel, linVelFrameLocal);

            var expectedYawRate = WrapPi(YawOf(nextFrameRot) - YawOf(frameRot)) / FrameTime;
            Assert.AreEqual(expectedYawRate, yawRate, Tolerance);
        }
        finally
        {
            nextPose.Dispose();
        }
    }

    [Test]
    public void ComputeVelocity_PureYawOfANonRootReferenceBone_ReturnsThatRate()
    {
        const float rate = 1.3f;

        var rotations = _pose.Rotations;
        var angularVelocities = _pose.AngularVelocities;
        SetRoot(_pose, new float3(0f, 1f, 0f), quaternion.RotateY(math.radians(35f)));
        rotations[1] = quaternion.RotateY(math.radians(10f));
        // Bone 1's local rate lives in bone 0's frame, which is a pure yaw, so it stays a pure yaw
        // in character space and the whole chain below turns with it.
        angularVelocities[1] = new float3(0f, rate, 0f);

        var def = new SimulationFrameDef
        {
            ReferenceBoneIndex = 2,
            ForwardAxisLocal = _skeleton.RestLocalAxis(2, math.forward())
        };

        SimulationFrame.ComputeVelocity(_pose, _skeletonData, def, FrameTime, out _, out var yawRate);

        Assert.AreEqual(rate, yawRate, Tolerance);
    }

    [Test]
    public void Compute_PoseWhoseRootIsFrameLocal_DerivesTheIdentityFrame()
    {
        SetRoot(_pose, new float3(-4f, 0.95f, 7f),
            math.mul(quaternion.RotateY(math.radians(115f)), quaternion.RotateZ(math.radians(12f))));

        var def = SimulationFrameDef.Default(_skeleton);
        SimulationFrame.Compute(_pose, _skeletonData, def, out var framePos, out var frameRot);

        var positions = _pose.Positions;
        var rotations = _pose.Rotations;
        positions[0] = SimulationFrame.ToFrameLocal(positions[0], framePos, frameRot);
        rotations[0] = SimulationFrame.ToFrameLocal(rotations[0], frameRot);

        SimulationFrame.Compute(_pose, _skeletonData, def, out var localFramePos, out var localFrameRot);

        AssertApprox(float3.zero, localFramePos);
        AssertApprox(quaternion.identity, localFrameRot);
    }

    [Test]
    public void FromFrameLocal_InvertsToFrameLocal()
    {
        var framePos = new float3(2f, 0f, -3f);
        var frameRot = quaternion.RotateY(math.radians(70f));
        var worldPos = new float3(2.4f, 1.1f, -2.2f);
        var worldRot = math.mul(quaternion.RotateY(math.radians(50f)), quaternion.RotateX(math.radians(8f)));

        AssertApprox(worldPos,
            SimulationFrame.FromFrameLocal(SimulationFrame.ToFrameLocal(worldPos, framePos, frameRot), framePos, frameRot));
        AssertApprox(worldRot,
            SimulationFrame.FromFrameLocal(SimulationFrame.ToFrameLocal(worldRot, frameRot), frameRot));
    }

    [Test]
    public void RecomposeRootVelocity_InvertsDecomposeRootVelocity()
    {
        var framePos = new float3(1f, 0f, -2f);
        var frameRot = quaternion.RotateY(math.radians(25f));
        var linVelFrameLocal = new float3(0.2f, 0f, 1.4f);
        const float yawRate = 0.75f;

        var rootPosition = new float3(1.1f, 0.98f, -1.9f);
        var rootVelocity = new float3(0.6f, -0.1f, 1.2f);
        var rootAngularVelocity = new float3(0.3f, 0.9f, -0.2f);

        SimulationFrame.DecomposeRootVelocity(framePos, frameRot, linVelFrameLocal, yawRate,
            rootPosition, rootVelocity, rootAngularVelocity, out var linVelLocal, out var angVelLocal);
        SimulationFrame.RecomposeRootVelocity(framePos, frameRot, linVelFrameLocal, yawRate,
            rootPosition, linVelLocal, angVelLocal, out var roundTripVelocity, out var roundTripAngularVelocity);

        AssertApprox(rootVelocity, roundTripVelocity);
        AssertApprox(rootAngularVelocity, roundTripAngularVelocity);
    }

    /// <summary>
    /// A root moving with the frame and nothing else contributes no leftover rate: decomposing a
    /// bone 0 that is rigidly carried by the frame gives zero on both channels.
    /// </summary>
    [Test]
    public void DecomposeRootVelocity_RootRigidlyCarriedByTheFrame_LeavesNothingBehind()
    {
        var framePos = new float3(0f, 0f, 0f);
        var frameRot = quaternion.identity;
        var linVelFrameLocal = new float3(0f, 0f, 2f);
        const float yawRate = 0.5f;

        var rootPosition = new float3(0f, 1f, 0f);
        var rootVelocity = linVelFrameLocal + math.cross(yawRate * math.up(), rootPosition - framePos);
        var rootAngularVelocity = yawRate * math.up();

        SimulationFrame.DecomposeRootVelocity(framePos, frameRot, linVelFrameLocal, yawRate,
            rootPosition, rootVelocity, rootAngularVelocity, out var linVelLocal, out var angVelLocal);

        AssertApprox(float3.zero, linVelLocal);
        AssertApprox(float3.zero, angVelLocal);
    }

    [Test]
    public void CharacterAngularVelocity_ChainOfLocalRates_SumsThroughAncestorRotations()
    {
        var rootRotation = quaternion.RotateY(math.radians(30f));
        var spineRotation = quaternion.RotateX(math.radians(45f));

        var rotations = _pose.Rotations;
        var angularVelocities = _pose.AngularVelocities;
        rotations[0] = rootRotation;
        rotations[1] = spineRotation;
        angularVelocities[0] = new float3(0.1f, 0.2f, 0.3f);
        angularVelocities[1] = new float3(0.4f, 0f, -0.2f);
        angularVelocities[2] = new float3(0f, -0.5f, 0.1f);

        // Each bone's rate is expressed in its parent's frame, so it reaches character space
        // through the rotations above that parent.
        var expected = angularVelocities[0]
                       + math.rotate(rootRotation, angularVelocities[1])
                       + math.rotate(math.mul(rootRotation, spineRotation), angularVelocities[2]);

        AssertApprox(expected, _skeletonData.CharacterSpaceAngularVelocity(_pose, 2));
        AssertApprox(angularVelocities[0], _skeletonData.CharacterSpaceAngularVelocity(_pose, 0));
    }
}
}
