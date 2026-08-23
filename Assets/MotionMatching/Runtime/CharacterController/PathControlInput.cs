using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Walks a closed polyline of waypoints. The straight-line counterpart to
/// <see cref="SplineControlInput"/>, with a per-segment speed, which makes it the easy way to script
/// a route with deliberate speed changes.
/// </summary>
/// <remarks>
/// Waypoints are in local XZ, so moving the GameObject moves the route. The last connects back to
/// the first.
/// </remarks>
public class PathControlInput : MotionMatchingControlInput
{
    public string trajectoryPositionFeatureName = "FuturePosition";
    public string trajectoryDirectionFeatureName = "FutureDirection";

    [Tooltip("Waypoints of the route, in local XZ. Wraps from the last back to the first.")]
    public KeyPoint[] path;

    // Position along the route, split into "which segment" and "how far along it".
    private int _currentKeyPoint;

    /// <summary>[0..1] which part of the current keypoint is the character currently at</summary>
    private float _currentKeyPointT;

    private float2 _currentPosition;
    private float2 _currentDirection;
    private float2[] _predictedPositions;
    private float2[] _predictedDirections;

    /// <summary>Speed of the segment currently being walked. Set as a side effect of
    /// <see cref="SimulatePath"/>, so it reflects the most recent call.</summary>
    private float _targetVelocity;

    // Features -----------------------------------------------------------------
    private int _trajectoryPosFeatureIndex;
    private int _trajectoryRotFeatureIndex;
    private int[] _trajectoryPosPredictionFrames;
    private int[] _trajectoryRotPredictionFrames;

    private int NumberPredictionPos
    {
        get { return _trajectoryPosPredictionFrames.Length; }
    }

    private int NumberPredictionRot
    {
        get { return _trajectoryRotPredictionFrames.Length; }
    }
    // --------------------------------------------------------------------------

    private void Start()
    {
        // Get the feature indices
        _trajectoryPosFeatureIndex = -1;
        _trajectoryRotFeatureIndex = -1;
        for (int i = 0; i < motionSynthesizer.GetMmData().trajectoryFeatures.Count; ++i)
        {
            if (motionSynthesizer.GetMmData().trajectoryFeatures[i].name == trajectoryPositionFeatureName)
                _trajectoryPosFeatureIndex = i;
            if (motionSynthesizer.GetMmData().trajectoryFeatures[i].name == trajectoryDirectionFeatureName)
                _trajectoryRotFeatureIndex = i;
        }

        Debug.Assert(_trajectoryPosFeatureIndex != -1, "Trajectory Position Feature not found");
        Debug.Assert(_trajectoryRotFeatureIndex != -1, "Trajectory Direction Feature not found");

        _trajectoryPosPredictionFrames = motionSynthesizer.GetMmData().trajectoryFeatures[_trajectoryPosFeatureIndex]
            .predictionFrames;
        _trajectoryRotPredictionFrames = motionSynthesizer.GetMmData().trajectoryFeatures[_trajectoryRotFeatureIndex]
            .predictionFrames;
        // TODO: generalize this, allow for different number of prediction frames
        Debug.Assert(_trajectoryPosPredictionFrames.Length == _trajectoryRotPredictionFrames.Length,
            "Trajectory Position and Trajectory Direction Prediction Frames must be the same for PathCharacterController");
        for (int i = 0; i < _trajectoryPosPredictionFrames.Length; ++i)
        {
            Debug.Assert(_trajectoryPosPredictionFrames[i] == _trajectoryRotPredictionFrames[i],
                "Trajectory Position and Trajectory Direction Prediction Frames must be the same for PathCharacterController");
        }

        _predictedPositions = new float2[NumberPredictionPos];
        _predictedDirections = new float2[NumberPredictionRot];
    }

    protected override void OnUpdate()
    {
        // Predict the future positions and directions
        for (int i = 0; i < NumberPredictionPos; i++)
        {
            SimulatePath(DatabaseDeltaTime * _trajectoryPosPredictionFrames[i], _currentKeyPoint, _currentKeyPointT,
                out _, out _,
                out _predictedPositions[i], out _predictedDirections[i]);
        }

        // Update Current Position and Direction
        SimulatePath(Time.deltaTime, _currentKeyPoint, _currentKeyPointT,
            out _currentKeyPoint, out _currentKeyPointT,
            out _currentPosition, out _currentDirection);
    }

    /// <summary>
    /// Walks forward by <paramref name="remainingTime"/> seconds from a given point and reports where
    /// that lands. Does not mutate route state, so the same call both advances the character and
    /// looks ahead to a prediction horizon.
    /// </summary>
    /// <remarks>
    /// Time is consumed segment by segment, since each can have its own speed and a long lookahead
    /// may cross several.
    /// </remarks>
    private void SimulatePath(float remainingTime, int currentKeypoint, float currentKeyPointT,
        out int nextKeypoint, out float nextKeyPointTime,
        out float2 nextPos, out float2 nextDir)
    {
        // Just in case remainingTime is negative or 0
        nextPos = float2.zero;
        nextDir = float2.zero;
        if (remainingTime <= 0)
        {
            KeyPoint current = path[currentKeypoint];
            KeyPoint next = path[(currentKeypoint + 1) % path.Length];
            float2 dir = next.Position - current.Position;
            nextPos = current.Position + dir * currentKeyPointT;
            nextDir = math.normalize(dir);
        }

        // Loop until the character has moved enough
        while (remainingTime > 0)
        {
            KeyPoint current = path[currentKeypoint];
            _targetVelocity = current.Velocity;
            KeyPoint next = path[(currentKeypoint + 1) % path.Length];
            float2 dir = next.Position - current.Position;
            float2 dirNorm = math.normalize(dir);
            float2 currentPos = current.Position + dir * currentKeyPointT;
            float timeToNext =
                math.distance(currentPos, next.Position) / _targetVelocity; // Time needed to get to the next keypoint
            float dt = math.min(remainingTime, timeToNext);
            remainingTime -= dt;
            if (remainingTime <= 0)
            {
                // Move
                currentPos += dirNorm * _targetVelocity * dt;
                currentKeyPointT = math.distance(current.Position, currentPos) /
                                   math.distance(current.Position, next.Position);
                nextPos = currentPos;
                nextDir = dirNorm;
            }
            else
            {
                // Advance to next keypoint
                currentKeypoint = (currentKeypoint + 1) % path.Length;
                currentKeyPointT = 0;
            }
        }

        Debug.Assert(math.abs(remainingTime) < 0.0001f,
            "Character did not move enough or moved to much. remainingTime = " + remainingTime);
        nextKeypoint = currentKeypoint;
        nextKeyPointTime = currentKeyPointT;
    }

    public quaternion GetCurrentRotation()
    {
        Quaternion rot = Quaternion.LookRotation(new Vector3(_currentDirection.x, 0, _currentDirection.y));
        return rot * transform.rotation;
    }

    public override void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> span)
    {
        if (!feature.simulationBone) Debug.Assert(false, "Trajectory should be computed using the simulation frame");
        switch (feature.featureType)
        {
            case TrajectoryFeatureChannel.Type.Position:
                float2 world = GetWorldPredictedPos(index);
                float3 local = character.InverseTransformPoint(new float3(world.x, 0.0f, world.y));
                span[0] = local.x;
                span[1] = local.z;
                break;
            case TrajectoryFeatureChannel.Type.Direction:
                float2 worldDir = GetWorldPredictedDir(index);
                float3 localDir = character.InverseTransformDirection(new Vector3(worldDir.x, 0.0f, worldDir.y));
                span[0] = localDir.x;
                span[1] = localDir.z;
                break;
            default:
                Debug.Assert(false, "Unknown feature type: " + feature.featureType);
                break;
        }
    }

    private float2 GetWorldPredictedPos(int index)
    {
        return _predictedPositions[index] + new float2(transform.position.x, transform.position.z);
    }

    private float2 GetWorldPredictedDir(int index)
    {
        return _predictedDirections[index];
    }

    public override float3 GetWorldInitPosition()
    {
        return new float3(path[0].Position.x, 0, path[0].Position.y) + (float3)transform.position;
    }

    public override float3 GetWorldInitDirection()
    {
        float2 dir = path.Length > 0 ? path[1].Position - path[0].Position : new float2(0, 1);
        return math.normalize(new float3(dir.x, 0, dir.y));
    }

    public override float3 GetPosition()
    {
        return transform.position + new Vector3(_currentPosition.x, 0, _currentPosition.y);
    }

    public override float GetTargetSpeed()
    {
        return _targetVelocity;
    }

    /// <summary>One waypoint of the route, and the speed to travel the segment that starts at it.</summary>
    [Serializable]
    public struct KeyPoint
    {
        /// <summary>Waypoint in the component's local XZ plane.</summary>
        public float2 Position;

        /// <summary>Speed in m/s along the segment from this waypoint to the next. A scalar, not a vector.</summary>
        public float Velocity;

        public float3 GetWorldPosition(Transform transform)
        {
            return transform.position + new Vector3(Position.x, 0, Position.y);
        }
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (path == null) return;

        const float heightOffset = 0.01f;

        // Draw KeyPoints
        Gizmos.color = Color.red;
        for (int i = 0; i < path.Length; i++)
        {
            float3 pos = path[i].GetWorldPosition(transform);
            Gizmos.DrawSphere(new Vector3(pos.x, heightOffset, pos.z), 0.1f);
        }

        // Draw path
        Gizmos.color = new Color(0.5f, 0.0f, 0.0f, 1.0f);
        for (int i = 0; i < path.Length - 1; i++)
        {
            float3 pos = path[i].GetWorldPosition(transform);
            float3 nextPos = path[i + 1].GetWorldPosition(transform);
            GizmosExtensions.DrawLine(new Vector3(pos.x, heightOffset, pos.z),
                new Vector3(nextPos.x, heightOffset, nextPos.z), 6);
        }

        // Last Line
        float3 lastPos = path[path.Length - 1].GetWorldPosition(transform);
        float3 firstPos = path[0].GetWorldPosition(transform);
        GizmosExtensions.DrawLine(new Vector3(lastPos.x, heightOffset, lastPos.z),
            new Vector3(firstPos.x, heightOffset, firstPos.z), 6);
        // Draw Velocity
        for (int i = 0; i < path.Length - 1; i++)
        {
            float3 pos = path[i].GetWorldPosition(transform);
            float3 nextPos = path[i + 1].GetWorldPosition(transform);
            Vector3 start = new Vector3(pos.x, heightOffset, pos.z);
            Vector3 end = new Vector3(nextPos.x, heightOffset, nextPos.z);
            GizmosExtensions.DrawArrow(start,
                start + (end - start).normalized * math.min(path[i].Velocity, math.distance(pos, nextPos)),
                thickness: 6);
        }

        // Last Line
        float3 lastPos2 = path[path.Length - 1].GetWorldPosition(transform);
        float3 firstPos2 = path[0].GetWorldPosition(transform);
        Vector3 start2 = new Vector3(lastPos2.x, heightOffset, lastPos2.z);
        Vector3 end2 = new Vector3(firstPos2.x, heightOffset, firstPos2.z);
        GizmosExtensions.DrawArrow(start2, start2 + (end2 - start2).normalized * path[path.Length - 1].Velocity,
            thickness: 3);

        // Draw Current Position And Direction
        if (!Application.isPlaying) return;
        Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
        Vector3 currentPos = (Vector3)GetPosition() + Vector3.up * heightOffset * 2;
        Gizmos.DrawSphere(currentPos, 0.1f);
        GizmosExtensions.DrawLine(currentPos, currentPos + (Quaternion)GetCurrentRotation() * Vector3.forward, 12);
        // Draw Prediction
        if (_predictedPositions == null || _predictedPositions.Length != NumberPredictionPos ||
            _predictedDirections == null || _predictedDirections.Length != NumberPredictionRot) return;
        Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
        for (int i = 0; i < NumberPredictionPos; i++)
        {
            float2 predictedPosf2 = GetWorldPredictedPos(i);
            Vector3 predictedPos = new Vector3(predictedPosf2.x, heightOffset * 2, predictedPosf2.y);
            Gizmos.DrawSphere(predictedPos, 0.1f);
            float2 dirf2 = GetWorldPredictedDir(i);
            GizmosExtensions.DrawLine(predictedPos, predictedPos + new Vector3(dirf2.x, 0.0f, dirf2.y) * 0.5f, 12);
        }
    }
#endif
}
}