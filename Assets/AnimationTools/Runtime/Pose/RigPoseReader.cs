using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Reads a live rig's Transforms into a <see cref="PoseBuffer"/> — the inverse of writing a
/// synthesized pose back onto them.
/// </summary>
/// <remarks>
/// Separate from <see cref="MotionSynthesisComponent"/>, which is its only caller today, because
/// the rates it derives are the one part of building a pose that has a right and a wrong answer,
/// and a MonoBehaviour method reached only from LateUpdate cannot be tested.
/// </remarks>
public static class RigPoseReader
{
    /// <summary>
    /// Overwrites <paramref name="pose"/> with where <paramref name="transforms"/> are now, and its
    /// velocity channels with the rates that carried the pose there.
    /// </summary>
    /// <param name="pose">
    /// The previous pose, read before being overwritten — so this is not idempotent, and calling it
    /// twice in one tick reports the second call's rates as zero.
    /// </param>
    /// <param name="transforms">The rig, indexed by skeleton bone index.</param>
    /// <param name="deltaTime">
    /// Length of the tick being closed. Zero or negative leaves the rates at zero rather than
    /// dividing by it, which is what a paused editor and the first tick both hand over.
    /// </param>
    /// <remarks>
    /// Bone 0 is read in world space and every other bone rotation-only, matching how a pose is
    /// stored: the bones below the root keep the rest offsets already in the buffer, which is what
    /// holds bone lengths fixed.
    /// <para>
    /// Rates are per second, and angular rates are rotation vectors in radians taken the short way
    /// round — the same quantities a database frame carries, so a stage that reads the incoming
    /// pose sees one kind of pose rather than two.
    /// </para>
    /// <para>
    /// Foot contacts are not recoverable from Transforms, so they are left untouched for whichever
    /// stage owns them.
    /// </para>
    /// </remarks>
    public static void Read(PoseBuffer pose, Transform[] transforms, float deltaTime)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;
        var velocities = pose.Velocities;
        var angularVelocities = pose.AngularVelocities;

        Debug.Assert(transforms.Length <= positions.Length,
            "The rig has more bones than the pose buffer has channels.");

        var inverseDeltaTime = deltaTime > 1e-9f ? 1f / deltaTime : 0f;

        var rootPosition = (float3)transforms[0].position;
        var rootRotation = (quaternion)transforms[0].rotation;
        velocities[0] = (rootPosition - positions[0]) * inverseDeltaTime;
        angularVelocities[0] = AngularRate(rotations[0], rootRotation, inverseDeltaTime);

        for (var i = 1; i < transforms.Length; i++)
        {
            var localPosition = (float3)transforms[i].localPosition;
            var localRotation = (quaternion)transforms[i].localRotation;
            velocities[i] = (localPosition - positions[i]) * inverseDeltaTime;
            angularVelocities[i] = AngularRate(rotations[i], localRotation, inverseDeltaTime);
        }

        positions[0] = rootPosition;
        rotations[0] = rootRotation;

        for (var i = 1; i < transforms.Length; i++)
        {
            rotations[i] = transforms[i].localRotation;
        }
    }

    /// <summary>
    /// Seeds <paramref name="pose"/> from <paramref name="transforms"/> with no rates at all, for
    /// the first pose of a run where there is no previous one to difference against.
    /// </summary>
    public static void Seed(PoseBuffer pose, Transform[] transforms)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;

        positions[0] = transforms[0].position;
        rotations[0] = transforms[0].rotation;

        for (var i = 1; i < transforms.Length; i++)
        {
            positions[i] = transforms[i].localPosition;
            rotations[i] = transforms[i].localRotation;
        }
    }

    /// <summary>
    /// Per-second rotation rate carrying <paramref name="from"/> to <paramref name="to"/>, as a
    /// rotation vector in radians taken the short way round.
    /// </summary>
    /// <remarks>
    /// Takes the reciprocal timestep rather than the timestep, so a zero-length tick yields a zero
    /// rate instead of a division by zero.
    /// </remarks>
    private static float3 AngularRate(quaternion from, quaternion to, float inverseDeltaTime)
    {
        var delta = MathExtensions.Abs(math.mul(to, math.inverse(from)));
        return MathExtensions.QuaternionToScaledAngleAxis(delta) * inverseDeltaTime;
    }
}
}
