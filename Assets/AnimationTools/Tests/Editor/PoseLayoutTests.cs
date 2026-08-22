using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using AnimationTools;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.TestTools;

namespace AnimationTools.Tests
{
public class PoseLayoutTests
{
    private sealed class TestWideChannel : ChannelDescriptor
    {
        private readonly int _id;

        public TestWideChannel(int id)
        {
            _id = id;
        }

        public override int FloatCount => 5;
        public override int SectionKey => ChannelSections.Bool + 1;

        public override bool Equals(ChannelDescriptor other)
        {
            return other is TestWideChannel wide && wide._id == _id;
        }

        public override int GetHashCode() => _id;

        public override int GetContentHash() => _id;
    }

    private Skeleton skeleton;

    [SetUp]
    public void SetUp()
    {
        skeleton = TestSkeletons.CreateChain3();
    }

    [TearDown]
    public void TearDown()
    {
        TestSkeletons.DestroyAll();
    }


    // Positions: root, head (2). Rotations: root, spine, head, Quaternion (3). Scale: spine (1).
    // Velocities: root, head (2). AngularVelocity: spine, RotationVector (1). Bools: spine
    // Default, head Contact (2).
    private List<ChannelDescriptor> BuildMixedChannels(RotationRepresentation rotationRepresentation = RotationRepresentation.Quaternion)
    {
        return new List<ChannelDescriptor>
        {
            new PositionChannel(TestSkeletons.RootId),
            new PositionChannel(TestSkeletons.HeadId),
            new RotationChannel(TestSkeletons.RootId, rotationRepresentation),
            new RotationChannel(TestSkeletons.SpineId, rotationRepresentation),
            new RotationChannel(TestSkeletons.HeadId, rotationRepresentation),
            new ScaleChannel(TestSkeletons.SpineId),
            new VelocityChannel(TestSkeletons.RootId),
            new VelocityChannel(TestSkeletons.HeadId),
            new AngularVelocityChannel(TestSkeletons.SpineId),
            new BoolChannel(TestSkeletons.SpineId, ChannelUsage.Default),
            new BoolChannel(TestSkeletons.HeadId, ChannelUsage.Contact)
        };
    }

    [Test]
    public void Build_MixedChannelSet_SectionOffsetsMatchHandComputedLayout()
    {
        var layout = PoseLayout.Build(skeleton, BuildMixedChannels());
        var data = layout.Data;

        Assert.AreEqual(0, data.PositionStart);
        Assert.AreEqual(2, data.PositionCount);

        Assert.AreEqual(6, data.RotationStart);
        Assert.AreEqual(3, data.RotationCount);
        Assert.AreEqual(4, data.RotationStride);

        Assert.AreEqual(18, data.ScaleStart);
        Assert.AreEqual(1, data.ScaleCount);

        Assert.AreEqual(21, data.VelocityStart);
        Assert.AreEqual(2, data.VelocityCount);

        Assert.AreEqual(27, data.AngularVelocityStart);
        Assert.AreEqual(1, data.AngularVelocityCount);

        Assert.AreEqual(30, data.BoolStart);
        Assert.AreEqual(2, data.BoolCount);

        Assert.AreEqual(32, layout.FloatCount);
    }

    [Test]
    public void Build_RotationVectorRepresentation_ShrinksRotationStrideAndFloatCount()
    {
        var layout = PoseLayout.Build(skeleton, BuildMixedChannels(RotationRepresentation.RotationVector));
        var data = layout.Data;

        Assert.AreEqual(3, data.RotationStride);
        // Same channel counts as the Quaternion layout, but 3 fewer floats per rotation * 3 rotations.
        Assert.AreEqual(29, layout.FloatCount);
    }

    [Test]
    public void Build_UnknownBoneId_Throws()
    {
        var channels = new List<ChannelDescriptor> { new PositionChannel(999) };
        Assert.Throws<ArgumentException>(() => PoseLayout.Build(skeleton, channels));
    }

    [Test]
    public void Build_DuplicateTriple_Throws()
    {
        var channels = new List<ChannelDescriptor>
        {
            new PositionChannel(TestSkeletons.RootId, usage: ChannelUsage.Default),
            new PositionChannel(TestSkeletons.RootId, usage: ChannelUsage.Default)
        };
        Assert.Throws<ArgumentException>(() => PoseLayout.Build(skeleton, channels));
    }

    [Test]
    public void Build_MixedRotationRepresentations_Throws()
    {
        var channels = new List<ChannelDescriptor>
        {
            new RotationChannel(TestSkeletons.RootId, RotationRepresentation.Quaternion),
            new RotationChannel(TestSkeletons.SpineId, RotationRepresentation.RotationVector)
        };
        Assert.Throws<ArgumentException>(() => PoseLayout.Build(skeleton, channels));
    }

    [Test]
    public void TryBind_ReturnsDistinctSlotsThatRoundTripIndependently()
    {
        var layout = PoseLayout.Build(skeleton, BuildMixedChannels());

        Assert.IsTrue(layout.TryBindChannel(new PositionChannel(TestSkeletons.RootId), out var rootHandle));
        Assert.IsTrue(layout.TryBindChannel(new PositionChannel(TestSkeletons.HeadId), out var headHandle));

        var buffer = PoseBuffer.Allocate(layout, Allocator.Temp);
        try
        {
            var rootValue = new float3(1f, 2f, 3f);
            var headValue = new float3(4f, 5f, 6f);

            buffer.SetFloat3(rootHandle, rootValue);
            buffer.SetFloat3(headHandle, headValue);

            // If both handles resolved to the same element, the second write would clobber the first.
            Assert.AreEqual(rootValue, buffer.GetFloat3(rootHandle));
            Assert.AreEqual(headValue, buffer.GetFloat3(headHandle));

            // Declaration order (root, then head) fixes which section element each bone owns.
            Assert.AreEqual(TestSkeletons.RootId, ((BoneChannelDescriptor)layout.GetChannel(ChannelSections.Position, 0)).BoneId);
            Assert.AreEqual(TestSkeletons.HeadId, ((BoneChannelDescriptor)layout.GetChannel(ChannelSections.Position, 1)).BoneId);
        }
        finally
        {
            buffer.Dispose();
        }
    }

    [Test]
    public void TryBind_UndeclaredChannel_ReturnsFalseWithInvalidHandle()
    {
        var channels = new List<ChannelDescriptor> { new PositionChannel(TestSkeletons.RootId) };
        var layout = PoseLayout.Build(skeleton, channels);

        var found = layout.TryBindChannel(new ScaleChannel(TestSkeletons.RootId), out var handle);

        Assert.IsFalse(found);
        Assert.IsFalse(handle.IsValid);
    }

    [Test]
    public void Bind_UndeclaredChannel_Throws()
    {
        var channels = new List<ChannelDescriptor> { new PositionChannel(TestSkeletons.RootId) };
        var layout = PoseLayout.Build(skeleton, channels);

        Assert.Throws<ArgumentException>(() => layout.BindChannel(new ScaleChannel(TestSkeletons.RootId)));
    }

    [Test]
    public void TryBindBool_UsageDisambiguatesSameBone()
    {
        var channels = new List<ChannelDescriptor>
        {
            new BoolChannel(TestSkeletons.HeadId, ChannelUsage.Default),
            new BoolChannel(TestSkeletons.HeadId, ChannelUsage.Contact)
        };
        var layout = PoseLayout.Build(skeleton, channels);

        Assert.IsTrue(layout.TryBindChannel(new BoolChannel(TestSkeletons.HeadId, ChannelUsage.Default), out var defaultHandle));
        Assert.IsTrue(layout.TryBindChannel(new BoolChannel(TestSkeletons.HeadId, ChannelUsage.Contact), out var contactHandle));

        var buffer = PoseBuffer.Allocate(layout, Allocator.Temp);
        try
        {
            buffer.SetBool(defaultHandle, true);

            // Distinct elements: setting only the Default-usage bool must not affect Contact.
            Assert.IsTrue(buffer.GetBool(defaultHandle));
            Assert.IsFalse(buffer.GetBool(contactHandle));
        }
        finally
        {
            buffer.Dispose();
        }
    }

    [Test]
    public void CreateFullPose_PositionAndRotationCountsMatchBoneCount()
    {
        var layout = PoseLayout.CreateFullPose(skeleton, false, false);

        Assert.AreEqual(skeleton.BoneCount, layout.Data.PositionCount);
        Assert.AreEqual(skeleton.BoneCount, layout.Data.RotationCount);
        Assert.AreEqual(0, layout.Data.VelocityCount);
        Assert.AreEqual(0, layout.Data.AngularVelocityCount);
    }

    [Test]
    public void CreateFullPose_ElementIndexEqualsBoneIndex()
    {
        var layout = PoseLayout.CreateFullPose(skeleton, false, false);

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            var expectedBoneId = skeleton.GetBoneId(i);
            Assert.AreEqual(expectedBoneId, ((BoneChannelDescriptor)layout.GetChannel(ChannelSections.Position, i)).BoneId);
            Assert.AreEqual(expectedBoneId, ((BoneChannelDescriptor)layout.GetChannel(ChannelSections.Rotation, i)).BoneId);
        }
    }

    [Test]
    public void CreateFullPose_IncludeVelocities_AddsBoneAlignedVelocitySection()
    {
        var layout = PoseLayout.CreateFullPose(skeleton, true, false);

        Assert.AreEqual(skeleton.BoneCount, layout.Data.VelocityCount);
        Assert.AreEqual(0, layout.Data.AngularVelocityCount);
        for (var i = 0; i < skeleton.BoneCount; i++)
            Assert.AreEqual(skeleton.GetBoneId(i), ((BoneChannelDescriptor)layout.GetChannel(ChannelSections.Velocity, i)).BoneId);
    }

    [Test]
    public void CreateFullPose_IncludeAngularVelocities_AddsBoneAlignedAngularVelocitySection()
    {
        var layout = PoseLayout.CreateFullPose(skeleton, false, true);

        Assert.AreEqual(0, layout.Data.VelocityCount);
        Assert.AreEqual(skeleton.BoneCount, layout.Data.AngularVelocityCount);
        for (var i = 0; i < skeleton.BoneCount; i++)
            Assert.AreEqual(skeleton.GetBoneId(i), ((BoneChannelDescriptor)layout.GetChannel(ChannelSections.AngularVelocity, i)).BoneId);
    }

    [Test]
    public void LayoutHash_IdenticalInputs_ProduceIdenticalHashAndCachedInstance()
    {
        var layoutA = PoseLayout.Build(skeleton, BuildMixedChannels());
        var layoutB = PoseLayout.Build(skeleton, BuildMixedChannels());

        Assert.AreEqual(layoutA.LayoutHash, layoutB.LayoutHash);
        Assert.IsTrue(ReferenceEquals(layoutA, layoutB));
    }

    [Test]
    public void LayoutHash_DifferentChannelList_ProducesDifferentHash()
    {
        var layoutA = PoseLayout.Build(skeleton, BuildMixedChannels());
        var layoutC = PoseLayout.CreateFullPose(skeleton, false, false);

        Assert.AreNotEqual(layoutA.LayoutHash, layoutC.LayoutHash);
    }

    [Test]
    public void Build_NonBuiltInChannel_PlacesAfterBoolSectionAndRoundTripsThroughChannelHandle()
    {
        var channels = new List<ChannelDescriptor>
        {
            new BoolChannel(TestSkeletons.RootId, ChannelUsage.Default),
            new TestWideChannel(1)
        };
        var layout = PoseLayout.Build(skeleton, channels);

        Assert.IsTrue(layout.TryGetSectionRange(ChannelSections.Bool, out var boolStart, out var boolCount));
        Assert.AreEqual(boolStart + boolCount, layout.Data.ExtraStart);
        Assert.AreEqual(5, layout.Data.ExtraCount);

        var handle = layout.BindChannel(new TestWideChannel(1));
        Assert.AreEqual(5, handle.FloatCount);

        var buffer = PoseBuffer.Allocate(layout, Allocator.Temp);
        try
        {
            buffer.SetFloat(handle, 3, 7.5f);
            Assert.AreEqual(7.5f, buffer.GetFloat(handle, 3));
        }
        finally
        {
            buffer.Dispose();
        }
    }

    [Test]
    public void LayoutHash_ChannelsDifferingOnlyInSpace_ProduceDifferentHash()
    {
        var channelsA = new List<ChannelDescriptor> { new PositionChannel(TestSkeletons.RootId, ChannelSpace.ParentLocal) };
        var channelsB = new List<ChannelDescriptor> { new PositionChannel(TestSkeletons.RootId, ChannelSpace.World) };

        var layoutA = PoseLayout.Build(skeleton, channelsA);
        var layoutB = PoseLayout.Build(skeleton, channelsB);

        Assert.AreNotEqual(layoutA.LayoutHash, layoutB.LayoutHash);
    }

    [Test]
    public void Handle_UsedOnBufferFromDifferentLayout_TripsLayoutAssert()
    {
        var layoutA = PoseLayout.Build(skeleton, new List<ChannelDescriptor> { new PositionChannel(TestSkeletons.RootId) });
        var layoutB = PoseLayout.Build(skeleton, new List<ChannelDescriptor>
        {
            new PositionChannel(TestSkeletons.RootId),
            new PositionChannel(TestSkeletons.HeadId)
        });
        Assert.AreNotEqual(layoutA.LayoutHash, layoutB.LayoutHash);

        var handle = layoutA.BindChannel(new PositionChannel(TestSkeletons.RootId));
        var buffer = PoseBuffer.Allocate(layoutB, Allocator.Temp);
        try
        {
            LogAssert.Expect(LogType.Assert, new Regex("ChannelHandle was bound against a different layout"));
            buffer.GetFloat3(handle);
        }
        finally
        {
            buffer.Dispose();
        }
    }
}
}
