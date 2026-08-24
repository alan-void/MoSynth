using System.Collections.Generic;
using AnimationTools;
using MotionMatching;
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
/// Checks that extraction produced what you meant, before any synthesis is involved. It shows the
/// <em>features</em> as well as the poses, so a trajectory feature pointing the wrong way — the usual
/// cause of a search picking odd frames — is visible here and nowhere else.
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
        // PoseSet
        _poseSet = motionMatchingData.GetOrImportPoseSet();

        // FeatureSet
        _featureSet = motionMatchingData.GetOrImportFeatureSet();

        // Skeleton
        _skeletonTransforms = SkeletonRigBuilder.CreateHierarchyWithMap(_poseSet.Skeleton, transform);

        // FPS
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
            for (int i = 0; i < rotations.Length; i++)
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
            for (int i = 0; i < _skeletonTransforms.Length; i++)
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
        // Skeleton
        if (_skeletonTransforms == null || _poseSet == null) return;

        Gizmos.color = Color.red;
        for (int i = 1; i < _skeletonTransforms.Length; i++) // bone 0 has no parent bone to draw to
        {
            Transform t = _skeletonTransforms[i];
            GizmosExtensions.DrawLine(t.parent.position, t.position, 3);
        }

        if (!play) return;
        // Character
        int currentFrame = math.max(0, this.currentFrame - 1); // FeatureDebug increments CurrentFrame after update... OnDrawGizmos is called after update
        var pose = _poseSet.GetPoseBuffer(currentFrame);
        FeatureSet.GetWorldOriginCharacter(pose, _poseSet.Skeleton.GetSkeletonData(), _poseSet.SimulationFrame,
            out float3 characterOrigin, out float3 characterForward);
        Gizmos.color = new Color(1.0f, 0.0f, 0.5f, 1.0f);
        Gizmos.DrawSphere(characterOrigin, spheresRadius);
        GizmosExtensions.DrawArrow(characterOrigin, characterOrigin + characterForward, thickness: 3);

        // Forward Trajectory Direction Features
        Gizmos.color = Color.gray;
        for (int t = 0; t < motionMatchingData.trajectoryFeatures.Count; t++)
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

        // Contacts
        if (debugContacts)
        {
            if (!BoneNameConventions.TryFindContactBone(_poseSet.Skeleton, left: true, out var leftToesIndex))
            {
                Debug.LogWarning("[MotionMatchingDataVisualiser] Could not find a left contact bone; falling back to the skeleton root.");
                leftToesIndex = 0;
            }
            if (!BoneNameConventions.TryFindContactBone(_poseSet.Skeleton, left: false, out var rightToesIndex))
            {
                Debug.LogWarning("[MotionMatchingDataVisualiser] Could not find a right contact bone; falling back to the skeleton root.");
                rightToesIndex = 0;
            }
            Gizmos.color = Color.green;
            if (pose.GetBool(_poseSet.LeftFootContactHandle))
            {
                Gizmos.DrawSphere(_skeletonTransforms[leftToesIndex].position, spheresRadius);
            }
            if (pose.GetBool(_poseSet.RightFootContactHandle))
            {
                Gizmos.DrawSphere(_skeletonTransforms[rightToesIndex].position, spheresRadius);
            }
        }

        // Feature Set
        if (_featureSet == null) return;

        DrawFeatureGizmos(_featureSet, motionMatchingData, spheresRadius, currentFrame, characterOrigin, characterForward,
            _skeletonTransforms, _poseSet.Skeleton, Color.blue, debugPose: debugPose, debugTrajectory: debugTrajectory);
    }

    private static List<float3> _positionFeatures = new();
    public static void DrawFeatureGizmos(FeatureSet set, MotionMatchingData mmData, float spheresRadius, int currentFrame,
        float3 characterOrigin, float3 characterForward, Transform[] joints, Skeleton skeleton,
        Color trajectoryColor, bool debugPose = true, bool debugTrajectory = true)
    {
        if (!set.IsValidFeature(currentFrame)) return;

        quaternion characterRot = quaternion.LookRotation(characterForward, math.up());

        // TODO: find a better way to store this information
        _positionFeatures.Clear();

        // Trajectory Features ---------------------------------------------------------------------------
        // Collect simulation-frame Position predictions; Direction gizmos anchor their arrows to them
        for (int t = 0; t < mmData.trajectoryFeatures.Count; t++)
        {
            var trajectoryFeature = mmData.trajectoryFeatures[t];
            if (trajectoryFeature.simulationBone && trajectoryFeature.featureType == TrajectoryFeatureChannel.Type.Position)
            {
                for (int p = 0; p < trajectoryFeature.predictionFrames.Length; p++)
                {
                    float3 value = trajectoryFeature.Unpack(set, currentFrame, t, p);
                    value = characterOrigin + math.mul(characterRot, value);
                    _positionFeatures.Add(value);
                }
            }
        }
        // Draw Trajectory Features
        if (debugTrajectory)
        {
            for (int t = 0; t < mmData.trajectoryFeatures.Count; t++)
            {
                var trajectoryFeature = mmData.trajectoryFeatures[t];
                for (int p = 0; p < trajectoryFeature.predictionFrames.Length; p++)
                {
                    DrawTrajectoryPoint(trajectoryFeature, set, currentFrame, t, p, trajectoryColor, characterOrigin, characterForward,
                        characterRot, spheresRadius, joints, skeleton);
                }
            }
        }
        // Pose Features ---------------------------------------------------------------------------
        if (debugPose)
        {
            Gizmos.color = new Color(0.0f, 0.8f, 0.8f);
            for (int p = 0; p < mmData.poseFeatures.Count; p++)
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
        int t = trajectoryFeatureIndex;
        int p = predictionIndex;
        //Gizmos.color = trajectoryColor * (1.25f - (float)p / trajectoryFeature.FramesPrediction.Length);
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
                    //GizmosExtensions.DrawArrow(jointPos, jointPos + value, 0.1f, thickness: 3);
                    GizmosExtensions.DrawArrow(jointPos, jointPos + value * 0.4f, 0.15f, thickness: 4);
                    break;
            }
        }
    }
#endif
}
}
