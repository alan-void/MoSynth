using System.Collections.Generic;
using AnimationTools;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
public class PoseBufferTests
{
    private Skeleton skeleton;
    private PoseLayout layout;
    private PoseBuffer buffer;

    private int rootBoneId;
    private int spineBoneId;
    private int headBoneId;

    [SetUp]
    public void SetUp()
    {
        skeleton = TestSkeletons.CreateChain3();
        rootBoneId = TestSkeletons.RootId;
        spineBoneId = TestSkeletons.SpineId;
        headBoneId = TestSkeletons.HeadId;

        var channels = new List<ChannelDescriptor>
        {
            new PositionChannel(TestSkeletons.RootId),
            new RotationChannel(TestSkeletons.RootId, RotationRepresentation.Quaternion),
            new ScaleChannel(TestSkeletons.SpineId),
            new VelocityChannel(TestSkeletons.RootId),
            new AngularVelocityChannel(TestSkeletons.RootId),
            new BoolChannel(TestSkeletons.HeadId, ChannelUsage.Default)
        };
        layout = PoseLayout.Build(skeleton, channels);
        buffer = PoseBuffer.Allocate(layout, Allocator.Temp);
    }

    [TearDown]
    public void TearDown()
    {
        buffer.Dispose();
        TestSkeletons.DestroyAll();
    }

    [Test]
    public void Allocate_ZeroInitializesAndIsCreated()
    {
        Assert.IsTrue(buffer.IsCreated);
        for (var i = 0; i < buffer.Data.Length; i++)
            Assert.AreEqual(0f, buffer.Data[i]);
    }

    [Test]
    public void SetGet_Position_RoundTrips()
    {
        var handle = layout.BindChannel(new PositionChannel(rootBoneId));
        var value = new float3(1f, 2f, 3f);
        buffer.SetFloat3(handle, value);
        Assert.AreEqual(value, buffer.GetFloat3(handle));
    }

    [Test]
    public void SetGet_QuaternionRotation_RoundTrips()
    {
        var handle = layout.BindChannel(new RotationChannel(rootBoneId, RotationRepresentation.Quaternion));
        var value = quaternion.AxisAngle(math.normalize(new float3(0f, 1f, 0f)), 0.7f);
        buffer.SetQuaternion(handle, value);
        var result = buffer.GetQuaternion(handle);
        Assert.AreEqual(value.value.x, result.value.x, 1e-6f);
        Assert.AreEqual(value.value.y, result.value.y, 1e-6f);
        Assert.AreEqual(value.value.z, result.value.z, 1e-6f);
        Assert.AreEqual(value.value.w, result.value.w, 1e-6f);
    }

    [Test]
    public void SetGet_Scale_RoundTrips()
    {
        var handle = layout.BindChannel(new ScaleChannel(spineBoneId));
        var value = new float3(2f, 3f, 4f);
        buffer.SetFloat3(handle, value);
        Assert.AreEqual(value, buffer.GetFloat3(handle));
    }

    [Test]
    public void SetGet_Velocity_RoundTrips()
    {
        var handle = layout.BindChannel(new VelocityChannel(rootBoneId));
        var value = new float3(-1f, 0.5f, 2f);
        buffer.SetFloat3(handle, value);
        Assert.AreEqual(value, buffer.GetFloat3(handle));
    }

    [Test]
    public void SetGet_AngularVelocity_RoundTrips()
    {
        var handle = layout.BindChannel(new AngularVelocityChannel(rootBoneId));
        var value = new float3(0f, 0f, 1.5f);
        buffer.SetFloat3(handle, value);
        Assert.AreEqual(value, buffer.GetFloat3(handle));
    }

    [Test]
    public void SetGet_Bool_RoundTripsTrueAndFalse()
    {
        var handle = layout.BindChannel(new BoolChannel(headBoneId, ChannelUsage.Default));

        buffer.SetBool(handle, true);
        Assert.IsTrue(buffer.GetBool(handle));

        buffer.SetBool(handle, false);
        Assert.IsFalse(buffer.GetBool(handle));
    }

    [Test]
    public void Slices_LengthsMatchSectionCounts()
    {
        Assert.AreEqual(layout.Data.PositionCount, buffer.Positions.Length);
        Assert.AreEqual(layout.Data.RotationCount, buffer.Rotations.Length);
    }

    [Test]
    public void PositionSlice_AliasesSetPosition()
    {
        var handle = layout.BindChannel(new PositionChannel(rootBoneId));
        var value = new float3(5f, 6f, 7f);

        buffer.SetFloat3(handle, value);
        Assert.AreEqual(value, buffer.Positions[0]);

        var viaSlice = new float3(8f, 9f, 10f);
        var positions = buffer.Positions;
        positions[0] = viaSlice;
        Assert.AreEqual(viaSlice, buffer.GetFloat3(handle));
    }

    [Test]
    public void CopyFrom_CopiesAllData()
    {
        var positionHandle = layout.BindChannel(new PositionChannel(rootBoneId));
        var boolHandle = layout.BindChannel(new BoolChannel(headBoneId, ChannelUsage.Default));
        buffer.SetFloat3(positionHandle, new float3(1f, 2f, 3f));
        buffer.SetBool(boolHandle, true);

        var destination = PoseBuffer.Allocate(layout, Allocator.Temp);
        try
        {
            destination.CopyFrom(buffer);

            for (var i = 0; i < layout.FloatCount; i++)
                Assert.AreEqual(buffer.Data[i], destination.Data[i]);
        }
        finally
        {
            destination.Dispose();
        }
    }

    [Test]
    public void Lerp_Positions_ComputesMidpoint()
    {
        var handle = layout.BindChannel(new PositionChannel(rootBoneId));
        var a = PoseBuffer.Allocate(layout, Allocator.Temp);
        var b = PoseBuffer.Allocate(layout, Allocator.Temp);
        var destination = PoseBuffer.Allocate(layout, Allocator.Temp);
        try
        {
            a.SetFloat3(handle, new float3(0f, 0f, 0f));
            b.SetFloat3(handle, new float3(10f, 20f, 30f));

            PoseBuffer.Lerp(a, b, 0.5f, destination);

            Assert.AreEqual(new float3(5f, 10f, 15f), destination.GetFloat3(handle));
        }
        finally
        {
            a.Dispose();
            b.Dispose();
            destination.Dispose();
        }
    }

    [Test]
    public void Lerp_Rotation_FixesHemisphereAndStaysNearIdentityAtHalfway()
    {
        var handle = layout.BindChannel(new RotationChannel(rootBoneId, RotationRepresentation.Quaternion));
        var a = PoseBuffer.Allocate(layout, Allocator.Temp);
        var b = PoseBuffer.Allocate(layout, Allocator.Temp);
        var destination = PoseBuffer.Allocate(layout, Allocator.Temp);
        try
        {
            a.SetQuaternion(handle, quaternion.identity);
            // Same rotation, opposite sign: naive lerp without the hemisphere fix would cancel out.
            var negated = quaternion.identity;
            negated.value = -negated.value;
            b.SetQuaternion(handle, negated);

            PoseBuffer.Lerp(a, b, 0.5f, destination);
            var result = destination.GetQuaternion(handle);

            Assert.AreEqual(1f, math.abs(result.value.w), 1e-4f);
            Assert.AreEqual(1f, math.length(result.value), 1e-4f);
        }
        finally
        {
            a.Dispose();
            b.Dispose();
            destination.Dispose();
        }
    }

    [Test]
    public void Lerp_Bool_StepsAtHalfway()
    {
        var handle = layout.BindChannel(new BoolChannel(headBoneId, ChannelUsage.Default));
        var a = PoseBuffer.Allocate(layout, Allocator.Temp);
        var b = PoseBuffer.Allocate(layout, Allocator.Temp);
        var destination = PoseBuffer.Allocate(layout, Allocator.Temp);
        try
        {
            a.SetBool(handle, false);
            b.SetBool(handle, true);

            PoseBuffer.Lerp(a, b, 0.49f, destination);
            Assert.IsFalse(destination.GetBool(handle));

            PoseBuffer.Lerp(a, b, 0.5f, destination);
            Assert.IsTrue(destination.GetBool(handle));
        }
        finally
        {
            a.Dispose();
            b.Dispose();
            destination.Dispose();
        }
    }

    [Test]
    public void PoseBuffer_IsAViewStruct_CopyAliasesSameData()
    {
        var handle = layout.BindChannel(new BoolChannel(headBoneId, ChannelUsage.Default));
        var copy = buffer;

        // Regression pin: PoseBuffer copies must alias, not clone, so a by-value contact-write
        // (e.g. a helper taking PoseBuffer by value) is visible through the original reference.
        copy.SetBool(handle, true);

        Assert.IsTrue(buffer.GetBool(handle));
    }
}
}
