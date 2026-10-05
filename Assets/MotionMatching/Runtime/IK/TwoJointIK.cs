using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
    /// <summary>
    /// Analytic two-bone IK on scene Transforms: places a three-joint chain so its end lands on a
    /// target. The classic use is planting a foot on uneven ground after synthesis has produced a pose
    /// that floats or sinks. The maths is <see cref="TwoBoneIK"/>'s.
    /// </summary>
    public static class TwoJointIK
    {
        /// <summary>
        /// Rotates <paramref name="jointA"/> and <paramref name="jointB"/> so <paramref name="jointC"/>
        /// lands on <paramref name="targetPos"/> (for a leg: hip, knee, ankle). A target within a
        /// millimetre of C leaves the chain untouched.
        /// </summary>
        /// <param name="forward">
        /// Which way the middle joint bends when the chain is straight. A bent chain keeps bending in
        /// the plane it already lies in.
        /// </param>
        public static void Solve(float3 targetPos, Transform jointA, Transform jointB, Transform jointC, float3 forward)
        {
            if (math.lengthsq(targetPos - (float3)jointC.position) < 0.001f * 0.001f) return;

            TwoBoneIK.Solve(jointA.position, jointB.position, jointC.position, jointA.rotation, jointB.rotation,
                targetPos, forward, out var rotationA, out var rotationB);
            jointA.rotation = rotationA;
            jointB.rotation = rotationB;
        }
    }
}
