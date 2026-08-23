using System;
using AnimationTools;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
public class PoseSequenceTests
{
    private const int FrameCount = 3;

    private Skeleton skeleton;
    private PoseLayout layout;

    [SetUp]
    public void SetUp()
    {
        skeleton = TestSkeletons.CreateChain3();
        layout = PoseLayout.CreateFullPose(skeleton, false, false);
    }

    [TearDown]
    public void TearDown()
    {
        TestSkeletons.DestroyAll();
    }


    [Test]
    public void Allocate_HasExpectedLengthAndIsZeroInitialized()
    {
        var sequence = PoseSequence.Allocate(layout, FrameCount, Allocator.Temp);
        try
        {
            Assert.AreEqual(FrameCount * layout.FloatCount, sequence.Data.Length);
            for (var i = 0; i < sequence.Data.Length; i++)
                Assert.AreEqual(0f, sequence.Data[i]);
        }
        finally
        {
            sequence.Dispose();
        }
    }

    [Test]
    public void FromArray_RoundTripsThroughCopyTo()
    {
        var source = new float[FrameCount * layout.FloatCount];
        for (var i = 0; i < source.Length; i++) source[i] = i * 0.5f + 1f;

        var sequence = PoseSequence.FromArray(layout, source, Allocator.Temp);
        try
        {
            var destination = new float[source.Length];
            sequence.CopyTo(destination);

            for (var i = 0; i < source.Length; i++)
                Assert.AreEqual(source[i], destination[i]);
        }
        finally
        {
            sequence.Dispose();
        }
    }

    [Test]
    public void GetFrame_RotationWriteThroughSlice_AliasesUnderlyingData()
    {
        var sequence = PoseSequence.Allocate(layout, FrameCount, Allocator.Temp);
        try
        {
            var value = quaternion.AxisAngle(math.normalize(new float3(1f, 0f, 0f)), 0.9f);

            var frame1 = sequence.GetFrame(1);
            var rotations = frame1.Rotations;
            rotations[0] = value;

            var baseIndex = 1 * layout.FloatCount + layout.Data.RotationStart;
            Assert.AreEqual(value.value.x, sequence.Data[baseIndex], 1e-6f);
            Assert.AreEqual(value.value.y, sequence.Data[baseIndex + 1], 1e-6f);
            Assert.AreEqual(value.value.z, sequence.Data[baseIndex + 2], 1e-6f);
            Assert.AreEqual(value.value.w, sequence.Data[baseIndex + 3], 1e-6f);
        }
        finally
        {
            sequence.Dispose();
        }
    }

    [Test]
    public void GetFrame_InteropsWithPoseFkAndPoseBufferCopyFrom()
    {
        var sequence = PoseSequence.Allocate(layout, FrameCount, Allocator.Temp);
        var outPositions = new NativeArray<float3>(skeleton.BoneCount, Allocator.Temp);
        var outRotations = new NativeArray<quaternion>(skeleton.BoneCount, Allocator.Temp);
        try
        {
            var frame0 = sequence.GetFrame(0);
            for (var i = 0; i < skeleton.BoneCount; i++)
            {
                var bone = skeleton.GetBone(i);
                var boneId = skeleton.GetBoneId(i);
                frame0.SetFloat3(layout.BindChannel(new PositionChannel(boneId)), bone.RestLocalPosition);
                frame0.SetQuaternion(layout.BindChannel(new RotationChannel(boneId)), quaternion.identity);
            }

            var skeletonData = skeleton.GetSkeletonData();
            Assert.DoesNotThrow(() => skeletonData.LocalSpaceToCharacterSpace(sequence.GetFrame(0), outPositions, outRotations));

            var copyDestination = PoseBuffer.Allocate(layout, Allocator.Temp);
            try
            {
                Assert.DoesNotThrow(() => copyDestination.CopyFrom(sequence.GetFrame(0)));
            }
            finally
            {
                copyDestination.Dispose();
            }
        }
        finally
        {
            outPositions.Dispose();
            outRotations.Dispose();
            sequence.Dispose();
        }
    }

    [Test]
    public void GetFrame_NegativeIndex_Throws()
    {
        var sequence = PoseSequence.Allocate(layout, FrameCount, Allocator.Temp);
        try
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => sequence.GetFrame(-1));
        }
        finally
        {
            sequence.Dispose();
        }
    }

    [Test]
    public void GetFrame_IndexEqualToFrameCount_Throws()
    {
        var sequence = PoseSequence.Allocate(layout, FrameCount, Allocator.Temp);
        try
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => sequence.GetFrame(FrameCount));
        }
        finally
        {
            sequence.Dispose();
        }
    }

    [Test]
    public void Constructor_DataLengthNotMultipleOfFloatCount_Throws()
    {
        var badData = new NativeArray<float>(layout.FloatCount + 1, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => new PoseSequence(layout, badData));
    }

    [Test]
    public void FromArray_LengthNotMultipleOfFloatCount_Throws()
    {
        var badArray = new float[layout.FloatCount + 1];
        Assert.Throws<ArgumentException>(() => PoseSequence.FromArray(layout, badArray, Allocator.Temp));
    }

    [Test]
    public void GetFrame_AdjacentFrames_AreDisjoint()
    {
        var sequence = PoseSequence.Allocate(layout, FrameCount, Allocator.Temp);
        try
        {
            var frame0 = sequence.GetFrame(0);
            var frame0Data = frame0.Data;
            for (var i = 0; i < frame0Data.Length; i++) frame0Data[i] = 7f;

            var frame1 = sequence.GetFrame(1);
            for (var i = 0; i < frame1.Data.Length; i++)
                Assert.AreEqual(0f, frame1.Data[i]);
        }
        finally
        {
            sequence.Dispose();
        }
    }
}
}
