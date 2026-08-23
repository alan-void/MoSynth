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
/// <remarks>
/// Same test as <see cref="AnimationTools.PoseExtractor"/> uses when baking contacts — character
/// velocity of the bone against a threshold, over the same skeleton. What it sees is not the same
/// input, though: this is the pose the whole pipeline produced, after searching and blending, so it
/// answers "is this foot planted right now" rather than "which frames will the next bake mark".
/// </remarks>
[Serializable]
public class ContactVisualizerStage : MoSynthStage
{
    [Tooltip("Bones to test for contact. Typically toes or feet; any bone works.")]
    [SerializeField] private List<SkeletonBone> contactBones = new();

    [Tooltip("Character-space speed (m/s) below which a bone counts as planted.")]
    [SerializeField] private float contactVelocityThreshold = 0.15f;

    [Tooltip("Size of the debug cross drawn at each contact bone, in metres.")]
    [SerializeField] private float markerSize = 0.08f;

    private MotionSynthesisComponent _component;
    private Skeleton _skeleton;

    // A private copy of the pipeline pose, in a layout that adds one Bool channel per configured
    // contact bone. The pipeline's own layout only carries the two built-in foot contacts, and a
    // read-only stage must not change the layout everything else shares.
    private PoseBuffer _buffer;

    private SkeletonData _skeletonData;

    /// <summary>Set when Init could not bind, after which Apply does nothing.</summary>
    private bool _inert;

    // Parallel arrays: _contactHandles[k] addresses the contact bool for bone _contactBoneIndices[k].
    private ChannelHandle[] _contactHandles;
    private int[] _contactBoneIndices;

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
            var velocity = _skeletonData.CharacterSpaceVelocity(_buffer, _contactBoneIndices[k]);
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
