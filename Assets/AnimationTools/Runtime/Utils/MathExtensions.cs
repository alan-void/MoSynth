using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

namespace AnimationTools
{
    /// <summary>
    /// Rotation maths the pipeline needs that Unity.Mathematics does not provide, centred on moving
    /// between quaternions and their scaled-angle-axis (rotation vector) form.
    /// </summary>
    /// <remarks>
    /// Quaternions cannot be added or scaled meaningfully, so anything treating rotation as a
    /// <em>rate</em> or an <em>offset</em> — angular velocity, inertialization, spring damping —
    /// works in scaled-angle-axis form instead: a plain float3, axis times angle in radians, which
    /// does behave like a vector. <see cref="Log"/> and <see cref="Exp"/> convert between the two;
    /// everything else here builds on them.
    /// <a href="https://theorangeduck.com/page/exponential-map-angle-axis-angular-velocity">Derivations</a>.
    /// </remarks>
    public static class MathExtensions
    {
        /// <summary>
        /// Quaternion absolute forces the quaternion to take the shortest path
        /// </summary>
        /// <remarks>
        /// q and -q are the same rotation, but only one is the short way round. This picks that one,
        /// so a difference of two rotations does not come back as 350 degrees instead of 10.
        /// </remarks>
        public static quaternion Abs(quaternion q)
        {
            return q.value.w < 0.0f ? new quaternion(-q.value.x, -q.value.y, -q.value.z, -q.value.w) : q;
        }

        /// <summary>Quaternion to rotation vector: axis times angle in radians.</summary>
        public static float3 QuaternionToScaledAngleAxis(quaternion q, float eps = 1e-8f)
        {
            return 2.0f * Log(q, eps);
        }

        /// <summary>Rotation vector back to a quaternion. Inverse of <see cref="QuaternionToScaledAngleAxis"/>.</summary>
        public static quaternion QuaternionFromScaledAngleAxis(float3 angleAxis, float eps = 1e-8f)
        {
            return Exp(angleAxis * 0.5f, eps);
        }

        /// <summary>
        /// Angular velocity carrying <paramref name="current"/> to <paramref name="next"/> over
        /// <paramref name="dt"/>, in radians per second as a rotation vector.
        /// </summary>
        public static float3 AngularVelocity(quaternion current, quaternion next, float dt)
        {
            // Rln = Rotation from local to world next
            // Rlc = Rotation from local to world current
            // Rcn = Rotation from world current to world next (angular velocity if divided by dt)
            // Rln = Rcn * Rlc * vl <- where vl is a vector in local space
            // Rcn = Rln * Rlc^-1
            // IF quaternions are not normalized try: return QuaternionToScaledAngleAxis(math.normalizesafe(Abs(math.mul(next, math.inverse(current))))) / dt;
            return QuaternionToScaledAngleAxis(Abs(math.mul(next, math.inverse(current)))) / dt;
        }

        /// <summary>
        /// Quaternion logarithm: half the rotation vector. The small-angle branch avoids acos and the
        /// division by length, both ill-conditioned near identity, where it agrees anyway.
        /// </summary>
        public static float3 Log(quaternion q, float eps = 1e-8f)
        {
            float length = math.sqrt(q.value.x * q.value.x + q.value.y * q.value.y + q.value.z * q.value.z);
            if (length < eps)
            {
                return new float3(q.value.x, q.value.y, q.value.z);
            }
            else
            {
                float halfangle = math.acos(math.clamp(q.value.w, -1f, 1f));
                return halfangle * (new float3(q.value.x, q.value.y, q.value.z) / length);
            }
        }

        /// <summary>
        /// Quaternion exponential, inverse of <see cref="Log"/>. Below the small-angle threshold
        /// sin(x)/x is taken as 1 rather than divided out.
        /// </summary>
        public static quaternion Exp(float3 angleAxis, float eps = 1e-8f)
        {
            float halfangle = math.sqrt(angleAxis.x * angleAxis.x + angleAxis.y * angleAxis.y + angleAxis.z * angleAxis.z);
            if (halfangle < eps)
            {
                return math.normalize(new quaternion(angleAxis.x, angleAxis.y, angleAxis.z, 1f));
            }
            else
            {
                float c = math.cos(halfangle);
                float s = math.sin(halfangle) / halfangle;
                return new quaternion(s * angleAxis.x, s * angleAxis.y, s * angleAxis.z, c);
            }
        }
    }
}