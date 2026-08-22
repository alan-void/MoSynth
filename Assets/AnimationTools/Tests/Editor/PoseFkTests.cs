using AnimationTools;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
public class PoseFkTests
{
    private const float Tolerance = 1e-4f;

    private Skeleton skeleton;
    private PoseLayout layout;
    private SkeletonData skeletonData;
    private PoseBuffer buffer;

    [SetUp]
    public void SetUp()
    {
        skeleton = TestSkeletons.CreateChain3();
        layout = PoseLayout.CreateFullPose(skeleton, true, true);
        skeletonData = skeleton.GetSkeletonData();
        buffer = PoseBuffer.Allocate(layout, Allocator.Temp);

        // Rest pose: local position/rotation per bone from the skeleton's own rest pose,
        // identity rotation, zero velocity — tests override only what they need to isolate.
        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            var bone = skeleton.GetBone(i);
            var boneId = skeleton.GetBoneId(i);
            buffer.SetFloat3(layout.BindChannel(new PositionChannel(boneId)), bone.RestLocalPosition);
            buffer.SetQuaternion(layout.BindChannel(new RotationChannel(boneId)), quaternion.identity);
        }
    }

    [TearDown]
    public void TearDown()
    {
        buffer.Dispose();
        TestSkeletons.DestroyAll();
    }

    private int BoneId(int index) => skeleton.GetBoneId(index);

    private static void AssertApprox(float3 expected, float3 actual)
    {
        Assert.AreEqual(expected.x, actual.x, Tolerance);
        Assert.AreEqual(expected.y, actual.y, Tolerance);
        Assert.AreEqual(expected.z, actual.z, Tolerance);
    }

    private static void AssertApprox(quaternion expected, quaternion actual)
    {
        var expectedValue = expected.value;
        var actualValue = actual.value;
        // q and -q represent the same rotation; align hemispheres before comparing.
        if (math.dot(expectedValue, actualValue) < 0f) actualValue = -actualValue;

        Assert.AreEqual(expectedValue.x, actualValue.x, Tolerance);
        Assert.AreEqual(expectedValue.y, actualValue.y, Tolerance);
        Assert.AreEqual(expectedValue.z, actualValue.z, Tolerance);
        Assert.AreEqual(expectedValue.w, actualValue.w, Tolerance);
    }

    [Test]
    public void LocalToCharacter_StraightChainIdentityRotations_AccumulatesPositionsAlongY()
    {
        var outPositions = new NativeArray<float3>(skeleton.BoneCount, Allocator.Temp);
        var outRotations = new NativeArray<quaternion>(skeleton.BoneCount, Allocator.Temp);
        try
        {
            PoseFK.LocalToCharacter(buffer, skeletonData, outPositions, outRotations);

            AssertApprox(new float3(0f, 0f, 0f), outPositions[0]);
            AssertApprox(new float3(0f, 1f, 0f), outPositions[1]);
            AssertApprox(new float3(0f, 2f, 0f), outPositions[2]);

            AssertApprox(quaternion.identity, outRotations[0]);
            AssertApprox(quaternion.identity, outRotations[1]);
            AssertApprox(quaternion.identity, outRotations[2]);
        }
        finally
        {
            outPositions.Dispose();
            outRotations.Dispose();
        }
    }

    [Test]
    public void LocalToCharacter_RootRotated90AboutZ_MatchesCharacterPositionAndRotationPerBone()
    {
        var rootRotation = quaternion.RotateZ(math.radians(90f));
        buffer.SetQuaternion(layout.BindChannel(new RotationChannel(BoneId(0))), rootRotation);

        var outPositions = new NativeArray<float3>(skeleton.BoneCount, Allocator.Temp);
        var outRotations = new NativeArray<quaternion>(skeleton.BoneCount, Allocator.Temp);
        try
        {
            PoseFK.LocalToCharacter(buffer, skeletonData, outPositions, outRotations);

            // Rotating (0,1,0) by 90 degrees about Z gives (-1,0,0); the chain accumulates it twice.
            AssertApprox(new float3(0f, 0f, 0f), outPositions[0]);
            AssertApprox(new float3(-1f, 0f, 0f), outPositions[1]);
            AssertApprox(new float3(-2f, 0f, 0f), outPositions[2]);

            for (var i = 0; i < skeleton.BoneCount; i++)
            {
                AssertApprox(outPositions[i], PoseFK.CharacterPosition(buffer, skeletonData, i));
                AssertApprox(outRotations[i], PoseFK.CharacterRotation(buffer, skeletonData, i));
            }
        }
        finally
        {
            outPositions.Dispose();
            outRotations.Dispose();
        }
    }

    [Test]
    public void CharacterPosition_MidBoneRotated90AboutX_LeafPositionMatchesHandComputed()
    {
        var spineRotation = quaternion.RotateX(math.radians(90f));
        buffer.SetQuaternion(layout.BindChannel(new RotationChannel(BoneId(1))), spineRotation);

        // Rotating the head's (0,1,0) offset by the spine's 90-degree X rotation gives (0,0,1).
        var expectedHeadPosition = new float3(0f, 1f, 1f);
        AssertApprox(expectedHeadPosition, PoseFK.CharacterPosition(buffer, skeletonData, 2));
    }

    [Test]
    public void CharacterVelocity_OnlyRootLinearVelocity_PropagatesUnchangedToEveryBone()
    {
        var rootVelocityHandle = layout.BindChannel(new VelocityChannel(BoneId(0)));
        buffer.SetFloat3(rootVelocityHandle, new float3(1f, 0f, 0f));

        for (var i = 0; i < skeleton.BoneCount; i++)
            AssertApprox(new float3(1f, 0f, 0f), PoseFK.CharacterVelocity(buffer, skeletonData, i));
    }

    [Test]
    public void CharacterVelocity_OnlyRootAngularVelocity_MatchesCrossProductWithLeverArm()
    {
        var rootAngularHandle = layout.BindChannel(new AngularVelocityChannel(BoneId(0)));
        buffer.SetFloat3(rootAngularHandle, new float3(0f, 0f, 1f));

        // Head sits at character-space (0,2,0) with all-identity rotations (see the straight-chain
        // position test); velocity of a point rigidly spun about the root is cross(w, r).
        var expected = math.cross(new float3(0f, 0f, 1f), new float3(0f, 2f, 0f));
        AssertApprox(expected, PoseFK.CharacterVelocity(buffer, skeletonData, 2));
    }

    [Test]
    public void CharacterVelocity_LeafLocalVelocity_RotatesThroughAncestorRotation()
    {
        var spineRotation = quaternion.RotateZ(math.radians(90f));
        buffer.SetQuaternion(layout.BindChannel(new RotationChannel(BoneId(1))), spineRotation);

        var headVelocityHandle = layout.BindChannel(new VelocityChannel(BoneId(2)));
        buffer.SetFloat3(headVelocityHandle, new float3(1f, 0f, 0f));

        // The head's local (1,0,0) velocity is carried into character space by the spine's
        // 90-degree Z rotation, which maps (1,0,0) -> (0,1,0).
        AssertApprox(new float3(0f, 1f, 0f), PoseFK.CharacterVelocity(buffer, skeletonData, 2));
    }
}
}
