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

        DrawFeatureGizmos(drawnFrame, characterOrigin, characterForward, Color.blue);
    }

    private FeatureVectorMarkers _featureMarkers;
    private SynthesisInputs _drawnFeatures;

    private static readonly Color PoseFeatureColor = new(0.0f, 0.8f, 0.8f);

    /// <summary>Draws one frame's stored features, decoded the same way a stage's recorded query is.</summary>
    private void DrawFeatureGizmos(int frame, float3 characterOrigin, float3 characterForward, Color trajectoryColor)
    {
        if (!_featureSet.IsValidFeature(frame)) return;

        if (_featureMarkers == null)
        {
            _featureMarkers = new FeatureVectorMarkers(motionMatchingData, _featureSet, _poseSet.Skeleton);
            _drawnFeatures = new SynthesisInputs("stored features", 256, 0);
        }

        var vector = _featureSet.GetFeatureVector(frame);
        var rotation = quaternion.LookRotation(characterForward, math.up());

        if (debugTrajectory)
        {
            _drawnFeatures.ClearMarkers();
            _featureMarkers.AddTrajectory(vector, default, characterOrigin, rotation, _skeletonTransforms,
                SynthesisInputRole.Answer, _drawnFeatures);
            DrawMarkers(_drawnFeatures, trajectoryColor);
        }

        if (debugPose)
        {
            _drawnFeatures.ClearMarkers();
            _featureMarkers.AddPose(vector, default, characterOrigin, rotation, _skeletonTransforms,
                SynthesisInputRole.Answer, _drawnFeatures);
            DrawMarkers(_drawnFeatures, PoseFeatureColor);
        }
    }

    private void DrawMarkers(SynthesisInputs markers, Color color)
    {
        Gizmos.color = color;
        for (var i = 0; i < markers.MarkerCount; i++)
        {
            ref readonly var marker = ref markers.GetMarker(i);
            if (!marker.IsArrow)
            {
                Gizmos.DrawSphere(marker.Position, spheresRadius);
                continue;
            }

            var length = math.length(marker.Vector);
            if (length < 0.001f) continue;
            GizmosExtensions.DrawArrow(marker.Position, marker.Position + marker.Vector, 0.3f * length, thickness: 4);
        }
    }

    private static int ResolveBoneOrZero(SkeletonBone bone, Skeleton skeleton, string featureName)
    {
        var index = bone?.ResolveIndex(skeleton) ?? -1;
        if (index >= 0) return index;

        Debug.LogWarning($"[MotionMatchingDataVisualiser] Feature \"{featureName}\" has no resolvable bone; falling back to the skeleton root.");
        return 0;
    }
#endif
}
}
