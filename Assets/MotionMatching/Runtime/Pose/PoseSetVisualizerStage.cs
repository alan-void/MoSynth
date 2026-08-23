using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Plays a pose database straight through, frame by frame and looping, with no search and no
/// control input. The way to look at what is actually in a database — to check an import, an
/// extraction or a retarget — with the rest of the pipeline out of the way.
/// </summary>
/// <remarks>
/// Like <see cref="MotionMatchingStage"/> this sources the pipeline skeleton from its database, so
/// it can stand in as the only stage on a component.
/// </remarks>
[Serializable]
public class PoseSetVisualizerStage : MoSynthStage
{
    private MotionSynthesisComponent _owner;

    /// <summary>The database to play back.</summary>
    public MotionMatchingData mmData;

    private PoseSet _poseSet;

    /// <summary>Frame of the database currently showing.</summary>
    public int CurrentFrame { get; private set; }

    /// <summary>Playhead as a float, so playback keeps database speed at any synthesis rate.</summary>
    private float _currentFrameTime;

    [Tooltip("Frame of the database to start playback from.")]
    [SerializeField]
    private int startFrame;


    public override Skeleton GetSkeleton(Skeleton inSkeleton)
    {
        _poseSet = mmData.GetOrImportPoseSet();
        return _poseSet.Skeleton;
    }
    
    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _owner = motionSynthesisComponent;
        _poseSet = mmData.GetOrImportPoseSet();
        CurrentFrame = startFrame;
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        // Advance frames with time
        _currentFrameTime = CurrentFrame + math.frac(_currentFrameTime);
        _currentFrameTime += deltaTime / _poseSet.FrameTime;
        CurrentFrame = (int)math.floor(_currentFrameTime);

        if (CurrentFrame >= _poseSet.NumberPoses)
        {
            CurrentFrame = 0;
        }
        pose.CopyFrom(_poseSet.GetPoseBuffer(CurrentFrame));
        return true;
    }
}
}