using System.Collections.Generic;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionMatching.Editor
{
/// <summary>
/// Draws a <see cref="MotionMatchingData"/>'s baked contents as gizmos: the poses and the trajectory
/// features extracted from them, played back frame by frame.
/// </summary>
/// <remarks>
/// Shows the features as well as the poses, so a trajectory feature pointing the wrong way (the
/// usual cause of a search picking odd frames) is visible before any synthesis is involved.
/// </remarks>
public class MotionMatchingDataVisualiser : MonoBehaviour
{
    public MotionMatchingData motionMatchingData;
    public bool play;
    public float spheresRadius = 0.05f;
    public bool lockFPS = true;
    public bool debugTrajectory = true;
    public bool debugPose = true;
    public bool debugContacts = true;

    private PoseSet _poseSet;
    private FeatureSet _featureSet;
    private Transform[] _skeletonTransforms;
    [SerializeField] private int currentFrame;

    private void Awake()
    {
        _poseSet = motionMatchingData.GetOrImportPoseSet();
        _featureSet = motionMatchingData.GetOrImportFeatureSet();
        _skeletonTransforms = SkeletonRigBuilder.CreateHierarchyWithMap(_poseSet.Skeleton, transform);

        if (lockFPS)
        {
            Application.targetFrameRate = Mathf.RoundToInt(1.0f / _poseSet.FrameTime);
            Debug.Log("[BVHDebug] Updated Target FPS: " + Application.targetFrameRate);
        }
        else
        {
            Application.targetFrameRate = -1;
        }
    }

    private void Update()
    {
        if (play)
        {
            var pose = _poseSet.GetPoseBuffer(currentFrame);
            var positions = pose.Positions;
            var rotations = pose.Rotations;
            _skeletonTransforms[0].localPosition = positions[0];
            for (var i = 0; i < rotations.Length; i++)
            {
                _skeletonTransforms[i].localRotation = rotations[i];
            }
            currentFrame = (currentFrame + 1) % _poseSet.NumberPoses;
        }
        else
        {
            // Parked at the origin in its rest pose, so the rig is readable while playback is off.
            currentFrame = 0;
            _skeletonTransforms[0].localPosition = float3.zero;
            for (var i = 0; i < _skeletonTransforms.Length; i++)
            {
                _skeletonTransforms[i].localRotation = quaternion.identity;
            }
        }
    }

    private void OnDestroy()
    {
        _poseSet.Dispose();
        _featureSet.Dispose();
    }

    private void OnApplicationQuit()
    {
        _poseSet.Dispose();
        _featureSet.Dispose();
    }


#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (_skeletonTransforms == null || _poseSet == null) return;

        Gizmos.color = Color.red;
        for (var i = 1; i < _skeletonTransforms.Length; i++) // bone 0 has no parent bone to draw to
        {
            var t = _skeletonTransforms[i];
            GizmosExtensions.DrawLine(t.parent.position, t.position, 3);
        }

        if (!play) return;
        // Update has already advanced currentFrame past the frame on screen.
        var drawnFrame = math.max(0, currentFrame - 1);
        var pose = _poseSet.GetPoseBuffer(drawnFrame);
        FeatureSet.GetWorldOriginCharacter(pose, _poseSet.Skeleton.GetSkeletonData(), _poseSet.SimulationFrame,
            out float3 characterOrigin, out float3 characterForward);
        Gizmos.color = new Color(1.0f, 0.0f, 0.5f, 1.0f);
        Gizmos.DrawSphere(characterOrigin, spheresRadius);
        GizmosExtensions.DrawArrow(characterOrigin, characterOrigin + characterForward, thickness: 3);

        Gizmos.color = Color.gray;
        for (var t = 0; t < motionMatchingData.trajectoryFeatures.Count; t++)
        {
            var trajectoryFeature = motionMatchingData.trajectoryFeatures[t];
            if (trajectoryFeature.featureType == TrajectoryFeatureChannel.Type.Direction &&
                !trajectoryFeature.simulationBone)
            {
                var jointIndex = ResolveBoneOrZero(trajectoryFeature.bone, _poseSet.Skeleton, trajectoryFeature.name);
                float3 dir = _skeletonTransforms[jointIndex].TransformDirection(motionMatchingData.GetLocalForward(jointIndex));
                float3 jointPos = _skeletonTransforms[jointIndex].position;
                GizmosExtensions.DrawArrow(jointPos, jointPos + dir * 0.5f, 0.1f, thickness: 3);
            }
        }

        if (debugContacts)
        {
            Gizmos.color = Color.green;
            var contacts = _poseSet.ContactHandles;
            for (var slot = 0; slot < contacts.Count; slot++)
            {
                if (!pose.GetBool(contacts[slot])) continue;

                Gizmos.DrawSphere(_skeletonTransforms[contacts.GetBoneIndex(slot)].position, spheresRadius);
            }
        }

        if (_featureSet == null) return;

        DrawFeatureGizmos(_featureSet, motionMatchingData, spheresRadius, drawnFrame, characterOrigin, characterForward,
            _skeletonTransforms, _poseSet.Skeleton, Color.blue, debugPose: debugPose, debugTrajectory: debugTrajectory);
    }

    // Simulation-frame position predictions of the frame being drawn; direction arrows anchor to them.
    private static readonly List<float3> _positionFeatures = new();

    public static void DrawFeatureGizmos(FeatureSet set, MotionMatchingData mmData, float spheresRadius, int currentFrame,
        float3 characterOrigin, float3 characterForward, Transform[] joints, Skeleton skeleton,
        Color trajectoryColor, bool debugPose = true, bool debugTrajectory = true)
    {
        if (!set.IsValidFeature(currentFrame)) return;

        quaternion characterRot = quaternion.LookRotation(characterForward, math.up());

        _positionFeatures.Clear();
        for (var t = 0; t < mmData.trajectoryFeatures.Count; t++)
        {
            var trajectoryFeature = mmData.trajectoryFeatures[t];
            if (trajectoryFeature.simulationBone && trajectoryFeature.featureType == TrajectoryFeatureChannel.Type.Position)
            {
                for (var p = 0; p < trajectoryFeature.predictionFrames.Length; p++)
                {
                    float3 value = trajectoryFeature.Unpack(set, currentFrame, t, p);
                    value = characterOrigin + math.mul(characterRot, value);
                    _positionFeatures.Add(value);
                }
            }
        }

        if (debugTrajectory)
        {
            for (var t = 0; t < mmData.trajectoryFeatures.Count; t++)
            {
                var trajectoryFeature = mmData.trajectoryFeatures[t];
                for (var p = 0; p < trajectoryFeature.predictionFrames.Length; p++)
                {
                    DrawTrajectoryPoint(trajectoryFeature, set, currentFrame, t, p, trajectoryColor, characterOrigin, characterForward,
                        characterRot, spheresRadius, joints, skeleton);
                }
            }
        }

        if (debugPose)
        {
            Gizmos.color = new Color(0.0f, 0.8f, 0.8f);
            for (var p = 0; p < mmData.poseFeatures.Count; p++)
            {
                var poseFeature = mmData.poseFeatures[p];
                float3 value = set.GetPoseFeature(currentFrame, p, true);
                switch (poseFeature.featureType)
                {
                    case PoseFeatureChannel.Type.Position:
                        value = characterOrigin + math.mul(characterRot, value);
                        Gizmos.DrawSphere(value, spheresRadius);
                        break;
                    case PoseFeatureChannel.Type.Velocity:
                        value = math.mul(characterRot, value);
                        if (math.length(value) > 0.001f)
                        {
                            var jointIndex = ResolveBoneOrZero(poseFeature.bone, skeleton, poseFeature.name);
                            float3 jointPos = joints[jointIndex].position;
                            GizmosExtensions.DrawArrow(jointPos, jointPos + value * 0.2f, 0.25f * math.length(value) * 0.2f, thickness: 4, useDepth: false);
                        }
                        break;
                }
            }
        }
    }

    private static int ResolveBoneOrZero(SkeletonBone bone, Skeleton skeleton, string featureName)
    {
        var index = bone?.ResolveIndex(skeleton) ?? -1;
        if (index >= 0) return index;

        Debug.LogWarning($"[MotionMatchingDataVisualiser] Feature \"{featureName}\" has no resolvable bone; falling back to the skeleton root.");
        return 0;
    }

    private static void DrawTrajectoryPoint(TrajectoryFeatureChannel trajectoryFeature, FeatureSet set, int currentFrame, int trajectoryFeatureIndex,
        int predictionIndex, Color trajectoryColor, float3 characterOrigin, float3 characterForward,
        quaternion characterRot, float spheresRadius, Transform[] joints, Skeleton skeleton)
    {
        var t = trajectoryFeatureIndex;
        var p = predictionIndex;
        Gizmos.color = trajectoryColor + (new Color(1.0f, 1.0f, 1.0f) - trajectoryColor) * ((float)p / trajectoryFeature.predictionFrames.Length);
        if (trajectoryFeature.featureType == TrajectoryFeatureChannel.Type.Position ||
            trajectoryFeature.featureType == TrajectoryFeatureChannel.Type.Direction)
        {
            float3 value = trajectoryFeature.Unpack(set, currentFrame, t, p);
            switch (trajectoryFeature.featureType)
            {
                case TrajectoryFeatureChannel.Type.Position:
                    value = characterOrigin + math.mul(characterRot, value);
                    Gizmos.DrawSphere(value, spheresRadius);
                    break;
                case TrajectoryFeatureChannel.Type.Direction:
                    float3 jointPos;
                    if (trajectoryFeature.simulationBone)
                    {
                        jointPos = _positionFeatures.Count > 0 ? _positionFeatures[p] : float3.zero;
                    }
                    else
                    {
                        var jointIndex = ResolveBoneOrZero(trajectoryFeature.bone, skeleton, trajectoryFeature.name);
                        jointPos = joints[jointIndex].position;
                    }
                    value = math.mul(characterRot, value);
                    GizmosExtensions.DrawArrow(jointPos, jointPos + value * 0.4f, 0.15f, thickness: 4);
                    break;
            }
        }
    }
#endif
}
}
