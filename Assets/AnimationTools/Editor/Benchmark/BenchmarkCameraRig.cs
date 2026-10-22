using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Places a benchmark video's camera: a damped chase from a 3/4 elevated angle, or one fixed view
/// framing the whole path.
/// </summary>
/// <remarks>
/// The chase follows the ground projection of the root through critically damped springs, so a
/// root snap (a motion matching jump, an exact root follow) eases into the shot instead of cutting it.
/// </remarks>
internal sealed class BenchmarkCameraRig
{
    public const float FieldOfView = 40f;

    private const float ChaseDistance = 5.5f;
    private const float ChasePitchDegrees = 20f;

    /// <summary>Orbit off straight-behind, which is what makes the view 3/4 rather than over-the-shoulder.</summary>
    private const float ChaseOrbitDegrees = 35f;

    private const float ChaseLookHeight = 0.9f;
    private const float PositionHalfLife = 0.3f;
    private const float YawHalfLife = 0.8f;
    private const float MaxYawRateDegrees = 75f;

    private const float OverviewPitchDegrees = 55f;
    private const float OverviewYawDegrees = 30f;
    private const float OverviewPadding = 1.5f;

    private const float Ln2 = 0.69314718f;

    private readonly Camera _camera;
    private readonly BenchmarkVideoCamera _mode;

    private float3 _focus;
    private float3 _focusVelocity;
    private float _yaw;
    private float _yawVelocity;
    private bool _hasState;

    public BenchmarkCameraRig(Camera camera, BenchmarkVideoCamera mode)
    {
        _camera = camera;
        _mode = mode;
        _camera.fieldOfView = FieldOfView;
    }

    /// <summary>How far the overview camera stands from the path's centre; 0 in chase mode.</summary>
    public float OverviewDistance { get; private set; }

    /// <summary>Forgets the chase state, so the next <see cref="Follow"/> snaps onto its target.</summary>
    public void Reset() => _hasState = false;

    /// <summary>Overview mode only: fixes the camera on a view that holds every point of the path.</summary>
    /// <param name="center">World centre of the path's ground-plane bounds.</param>
    /// <param name="radius">Ground-plane radius around <paramref name="center"/> enclosing the path.</param>
    public void FramePath(Vector3 center, float radius)
    {
        if (_mode != BenchmarkVideoCamera.Overview) return;

        // Fit the bounding sphere in the narrower of the two field-of-view axes.
        var halfVertical = math.radians(FieldOfView) * 0.5f;
        var halfHorizontal = math.atan(math.tan(halfVertical) * _camera.aspect);
        var halfAngle = math.min(halfVertical, halfHorizontal);
        OverviewDistance = (radius + OverviewPadding) / math.sin(halfAngle);

        var rotation = Quaternion.Euler(OverviewPitchDegrees, OverviewYawDegrees, 0f);
        var position = center - rotation * Vector3.forward * OverviewDistance;
        _camera.transform.SetPositionAndRotation(position, rotation);
    }

    /// <summary>Chase mode only: advances the springs one tick toward the character and places the camera.</summary>
    /// <param name="root">The character frame's world position.</param>
    /// <param name="forward">The character frame's world facing; only its ground-plane heading is used.</param>
    /// <param name="groundHeight">World height of the ground the character walks on.</param>
    public void Follow(Vector3 root, Vector3 forward, float groundHeight, float deltaTime)
    {
        if (_mode != BenchmarkVideoCamera.Chase) return;

        var goalFocus = new float3(root.x, groundHeight + ChaseLookHeight, root.z);
        var goalYaw = math.lengthsq(new float2(forward.x, forward.z)) > 1e-8f
            ? math.atan2(forward.x, forward.z)
            : _yaw;

        if (!_hasState)
        {
            _focus = goalFocus;
            _focusVelocity = float3.zero;
            _yaw = goalYaw;
            _yawVelocity = 0f;
            _hasState = true;
        }
        else if (deltaTime > 0f)
        {
            DampCritical(ref _focus, ref _focusVelocity, goalFocus, PositionHalfLife, deltaTime);

            // Unwrapped onto the current yaw so the spring always takes the short way round.
            var unwrappedGoal = _yaw + WrapAngle(goalYaw - _yaw);
            var previousYaw = _yaw;
            DampCritical(ref _yaw, ref _yawVelocity, unwrappedGoal, YawHalfLife, deltaTime);

            var maxStep = math.radians(MaxYawRateDegrees) * deltaTime;
            var step = _yaw - previousYaw;
            if (math.abs(step) > maxStep)
            {
                _yaw = previousYaw + math.sign(step) * maxStep;
                _yawVelocity = math.sign(step) * math.radians(MaxYawRateDegrees);
            }
        }

        var rotation = Quaternion.Euler(ChasePitchDegrees, math.degrees(_yaw) + ChaseOrbitDegrees, 0f);
        var position = (Vector3)_focus - rotation * Vector3.forward * ChaseDistance;
        _camera.transform.SetPositionAndRotation(position, rotation);
    }

    /// <summary>Exact critically damped spring step (Holden's damper): no overshoot at any time step.</summary>
    private static void DampCritical(ref float x, ref float v, float goal, float halfLife, float deltaTime)
    {
        var y = 2f * Ln2 / math.max(halfLife, 1e-5f);
        var j0 = x - goal;
        var j1 = v + j0 * y;
        var decay = math.exp(-y * deltaTime);
        x = decay * (j0 + j1 * deltaTime) + goal;
        v = decay * (v - j1 * y * deltaTime);
    }

    private static void DampCritical(ref float3 x, ref float3 v, float3 goal, float halfLife, float deltaTime)
    {
        var y = 2f * Ln2 / math.max(halfLife, 1e-5f);
        var j0 = x - goal;
        var j1 = v + j0 * y;
        var decay = math.exp(-y * deltaTime);
        x = decay * (j0 + j1 * deltaTime) + goal;
        v = decay * (v - j1 * y * deltaTime);
    }

    private static float WrapAngle(float radians) => radians - 2f * math.PI * math.round(radians / (2f * math.PI));
}
}
