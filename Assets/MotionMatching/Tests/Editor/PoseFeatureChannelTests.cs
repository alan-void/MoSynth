using AnimationTools;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching.Tests
{
/// <summary>
/// Extraction of the pose half of a feature vector. The velocity feature differences against the
/// next frame, so the interesting cases are all about which frame that is.
/// </summary>
public class PoseFeatureChannelTests
{
    private const float FrameTime = 1f / 30f;

    private Skeleton _skeleton;
    private MotionMatchingData _mmData;

    [SetUp]
    public void SetUp()
    {
        _skeleton = MmTestData.BuildSkeleton();
        _mmData = ScriptableObject.CreateInstance<MotionMatchingData>();
    }

    [TearDown]
    public void TearDown()
    {
        MmTestData.DestroyAll();
        Object.DestroyImmediate(_mmData);
    }

    /// <summary>
    /// Two clips of <paramref name="framesPerClip"/> frames each, whose roots march along +X at
    /// <paramref name="stepPerFrame"/> per frame. Clip 1 starts far away from where clip 0 ended,
    /// so reading across the boundary is unmistakable.
    /// </summary>
    private PoseSet BuildTwoClipSet(int framesPerClip, float stepPerFrame, float clipGap)
    {
        var poseSet = new PoseSet();
        poseSet.SetSkeleton(_skeleton);

        for (var clip = 0; clip < 2; clip++)
        {
            var frames = poseSet.BeginClip(framesPerClip, FrameTime);
            for (var i = 0; i < framesPerClip; i++)
            {
                var pose = frames[i];
                var positions = pose.Positions;
                var rotations = pose.Rotations;
                for (var bone = 0; bone < _skeleton.BoneCount; bone++)
                {
                    positions[bone] = bone == 0
                        ? new float3(clip * clipGap + i * stepPerFrame, 1f, 0f)
                        : _skeleton.GetBone(bone).RestLocalPosition;
                    rotations[bone] = quaternion.identity;
                }
            }
        }

        return poseSet;
    }

    private float3 ExtractVelocity(PoseSet poseSet, int poseIndex, int boneIndex)
    {
        var channel = new PoseFeatureChannel
        {
            name = "footVelocity",
            featureType = PoseFeatureChannel.Type.Velocity
        };

        var layout = PoseLayout.Build(_skeleton, new ChannelDescriptor[] { channel });
        var frame = StateBuffer.Allocate(layout, Allocator.Temp);
        try
        {
            var handle = layout.BindChannel(channel);
            channel.Extract(poseSet, _mmData, poseIndex, boneIndex, handle, frame);
            return frame.GetFloat3(handle);
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public void VelocityFeature_WithinAClip_DifferencesAgainstTheNextFrame()
    {
        var poseSet = BuildTwoClipSet(framesPerClip: 4, stepPerFrame: 0.1f, clipGap: 100f);

        // The whole rig translates rigidly, so every joint moves at the root's speed.
        var velocity = ExtractVelocity(poseSet, poseIndex: 1, boneIndex: 3);

        Assert.That(velocity.x, Is.EqualTo(0.1f / FrameTime).Within(1e-3f));
    }

    [Test]
    public void VelocityFeature_OnAClipsLastFrame_DoesNotReadTheNextClip()
    {
        const int framesPerClip = 4;
        var poseSet = BuildTwoClipSet(framesPerClip, stepPerFrame: 0.1f, clipGap: 100f);

        var velocity = ExtractVelocity(poseSet, poseIndex: framesPerClip - 1, boneIndex: 3);

        Assert.That(math.length(velocity), Is.EqualTo(0f).Within(1e-3f),
            "The frame after a clip's last belongs to a different animation; differencing against " +
            "it reports the jump-cut as joint velocity.");
    }
}
}
