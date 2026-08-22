using System;
using System.Collections.Generic;
using AnimationTools;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionMatching
{
/// <summary>
/// Read-only pass-through stage that estimates foot-style contact for a set of bones on the
/// pipeline skeleton and draws it as a gizmo-free debug marker. Never mutates the pose.
/// </summary>
[Serializable]
public class ContactVisualizerStage : MoSynthStage
{
    [SerializeField] private List<SkeletonBone> contactBones = new();
    [SerializeField] private float contactVelocityThreshold = 0.15f;
    [SerializeField] private float markerSize = 0.08f;

    private MotionSynthesisComponent _component;
    private Skeleton _skeleton;
    private PoseBuffer _buffer;
    private SkeletonData _skeletonData;
    private bool _inert;

    private ChannelHandle[] _contactHandles;
    private int[] _contactBoneIndices;

    public override Skeleton GetSkeleton(Skeleton inSkeleton) => inSkeleton;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _component = motionSynthesisComponent;
        _skeleton = motionSynthesisComponent.Skeleton;

        if (_skeleton == null)
        {
            Debug.LogWarning($"[ContactVisualizerStage] \"{motionSynthesisComponent.name}\": no pipeline skeleton available. Disabling.");
            _inert = true;
            return;
        }

        _skeletonData = _skeleton.GetSkeletonData();

        var extraChannels = new List<ChannelDescriptor>();
        foreach (var boneRef in contactBones)
        {
            if (boneRef == null || !boneRef.IsSet) continue;
            var boneIndex = boneRef.ResolveIndex(_skeleton);
            if (boneIndex < 0) continue;
            extraChannels.Add(new BoolChannel(_skeleton.GetBoneId(boneIndex), ChannelUsage.Contact));
        }

        var channels = PoseLayoutBuilder.BuildFullPoseChannels(_skeleton);
        channels.AddRange(extraChannels);
        var layout = PoseLayout.Build(_skeleton, channels);
        _buffer = PoseBuffer.Allocate(layout, Allocator.Persistent);

        var handles = new List<ChannelHandle>();
        var boneIndices = new List<int>();

        foreach (var boneRef in contactBones)
        {
            if (boneRef == null || !boneRef.IsSet) continue;

            var boneIndex = boneRef.ResolveIndex(_skeleton);
            if (boneIndex < 0)
            {
                Debug.LogWarning($"[ContactVisualizerStage] \"{motionSynthesisComponent.name}\": contact bone \"{boneRef.Name}\" not found in the pipeline skeleton; skipping.");
                continue;
            }

            handles.Add(layout.BindChannel(new BoolChannel(_skeleton.GetBoneId(boneIndex), ChannelUsage.Contact)));
            boneIndices.Add(boneIndex);
        }

        _contactHandles = handles.ToArray();
        _contactBoneIndices = boneIndices.ToArray();
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (_inert) return true;

        var srcPositions = pose.Positions;
        var srcRotations = pose.Rotations;
        var srcVelocities = pose.Velocities;
        var srcAngularVelocities = pose.AngularVelocities;
        var dstPositions = _buffer.Positions;
        var dstRotations = _buffer.Rotations;
        var dstVelocities = _buffer.Velocities;
        var dstAngularVelocities = _buffer.AngularVelocities;
        var boneCount = _skeleton.BoneCount;
        for (var i = 0; i < boneCount; i++)
        {
            dstPositions[i] = srcPositions[i];
            dstRotations[i] = srcRotations[i];
            dstVelocities[i] = srcVelocities[i];
            dstAngularVelocities[i] = srcAngularVelocities[i];
        }

        for (var k = 0; k < _contactHandles.Length; k++)
        {
            var velocity = PoseFK.CharacterVelocity(_buffer, _skeletonData, _contactBoneIndices[k]);
            var contact = math.length(velocity) < contactVelocityThreshold;
            _buffer.SetBool(_contactHandles[k], contact);

            var worldPosition = _component.SkeletonTransforms[_contactBoneIndices[k]].position;
            DrawContactMarker(worldPosition, contact);
        }

        return true;
    }

    private void DrawContactMarker(float3 position, bool contact)
    {
        var color = contact ? Color.green : Color.red;
        var half = markerSize * 0.5f;
        Debug.DrawLine(position - new float3(half, 0f, 0f), position + new float3(half, 0f, 0f), color, 0f);
        Debug.DrawLine(position - new float3(0f, half, 0f), position + new float3(0f, half, 0f), color, 0f);
        Debug.DrawLine(position - new float3(0f, 0f, half), position + new float3(0f, 0f, half), color, 0f);
    }

    public override void OnValidate()
    {
        contactVelocityThreshold = math.max(0f, contactVelocityThreshold);
        markerSize = math.max(0f, markerSize);
    }

    public override void OnDestroy()
    {
        if (_buffer.IsCreated) _buffer.Dispose();
    }
}
}
