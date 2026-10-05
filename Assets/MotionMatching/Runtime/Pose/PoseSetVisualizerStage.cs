using System;
using System.Collections.Generic;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Plays a pose database straight through, frame by frame and looping, with no search and no
/// control input, to inspect an import, extraction or retarget in isolation.
/// </summary>
/// <remarks>
/// The component's skeleton must be the database's own rig. It can be the only stage on a component.
/// </remarks>
[Serializable]
public class PoseSetVisualizerStage : MoSynthStage, IContactBoneSource
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

    public IReadOnlyList<string> ContactBoneNames => mmData == null ? null : mmData.ContactBoneNames;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _owner = motionSynthesisComponent;

        if (mmData == null)
        {
            Debug.LogError($"PoseSetVisualizerStage on \"{motionSynthesisComponent.name}\": no " +
                           "MotionMatchingData assigned.");
            return;
        }

        _poseSet = mmData.GetOrImportPoseSet();
        if (_poseSet == null)
        {
            mmData.TryValidate(out var error);
            Debug.LogError($"PoseSetVisualizerStage: MotionMatchingData \"{mmData.name}\" is not usable — {error}");
            return;
        }

        if (!Skeleton.StructurallyEqual(motionSynthesisComponent.Skeleton, _poseSet.Skeleton))
        {
            Debug.LogError($"PoseSetVisualizerStage on \"{motionSynthesisComponent.name}\": the component's " +
                           $"Skeleton is not the rig MotionMatchingData \"{mmData.name}\" was built over. " +
                           "Assign that rig's root bone, or regenerate the database.");
            _poseSet = null;
            return;
        }

        if (!ContactBoneSources.SameNames(_poseSet.ContactBoneNames, motionSynthesisComponent.ContactBoneNames))
        {
            Debug.LogError($"PoseSetVisualizerStage on \"{motionSynthesisComponent.name}\": the database flags " +
                           $"contacts on {ContactBoneSources.Describe(_poseSet.ContactBoneNames)} but the " +
                           $"component adopted {ContactBoneSources.Describe(motionSynthesisComponent.ContactBoneNames)}.");
            _poseSet = null;
            return;
        }

        CurrentFrame = startFrame;
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (_poseSet == null) return true;

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