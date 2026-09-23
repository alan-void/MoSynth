using Unity.Mathematics;

namespace AnimationTools
{
    /// <summary>
    /// Critically damped springs and dampers, from Daniel Holden's
    /// <a href="https://theorangeduck.com/page/spring-roll-call">spring roll call</a>; local names
    /// (y, j0, j1, eyedt) follow that article.
    /// </summary>
    public static class Spring
    {
        /// <summary>
        /// Variation of the damper code that damps a point starting at zero moving toward the desired difference
        /// </summary>
        public static float3 DampAdjustmentImplicit(float3 goal, float halfLife, float dt, float eps = 1e-5f)
        {
            const float ln2 = 0.69314718056f;
            return goal * (1.0f - FastNegativeExponent((ln2 * dt) / (halfLife + eps)));
        }
        /// <summary>
        /// Variation of the damper code that damps a point starting at zero moving toward the desired difference
        /// </summary>
        public static float2 DampAdjustmentImplicit(float2 goal, float halfLife, float dt, float eps = 1e-5f)
        {
            const float ln2 = 0.69314718056f;
            return goal * (1.0f - FastNegativeExponent((ln2 * dt) / (halfLife + eps)));
        }
        /// <summary>
        /// Variation of the damper code that damps a value starting at zero moving toward the desired difference
        /// </summary>
        public static float DampAdjustmentImplicit(float goal, float halfLife, float dt, float eps = 1e-5f)
        {
            const float ln2 = 0.69314718056f;
            return goal * (1.0f - FastNegativeExponent((ln2 * dt) / (halfLife + eps)));
        }
        /// <summary>
        /// Variation of the damper code that damps a rotation starting at the identity rotation toward the desired difference
        /// </summary>
        public static quaternion DampAdjustmentImplicit(quaternion goal, float halfLife, float dt, float eps = 1e-5f)
        {
            const float ln2 = 0.69314718056f;
            return math.slerp(quaternion.identity, goal, 1.0f - FastNegativeExponent((ln2 * dt) / (halfLife + eps)));
        }
        /// <summary>
        /// Given a position, velocity, desired velocity and acceleration, returns the new
        /// position, velocity and acceleration after deltaTime seconds.
        /// </summary>
        public static void CharacterPositionUpdate(ref float2 pos, ref float2 velocity, ref float2 acceleration,
                                                   float2 velocityGoal, float halfLife, float deltaTime)
        {
            var y = HalfLifeToDamping(halfLife) / 2.0f;
            var j0 = velocity - velocityGoal;
            var j1 = acceleration + j0 * y;
            var eyedt = FastNegativeExponent(y * deltaTime);

            pos = eyedt * (((-j1) / (y * y)) + ((-j0 - j1 * deltaTime) / y)) +
                  (j1 / (y * y)) + j0 / y + velocityGoal * deltaTime + pos;
            velocity = eyedt * (j0 + j1 * deltaTime) + velocityGoal;
            acceleration = eyedt * (acceleration - j1 * y * deltaTime);
        }
        /// <summary>
        /// Given a position, velocity, desired velocity and acceleration, returns the new
        /// position, velocity and acceleration after deltaTime seconds.
        /// </summary>
        public static void CharacterPositionUpdate(ref float3 pos, ref float3 velocity, ref float3 acceleration,
                                                   float3 velocityGoal, float halfLife, float deltaTime)
        {
            var y = HalfLifeToDamping(halfLife) / 2.0f;
            var j0 = velocity - velocityGoal;
            var j1 = acceleration + j0 * y;
            var eyedt = FastNegativeExponent(y * deltaTime);

            pos = eyedt * (((-j1) / (y * y)) + ((-j0 - j1 * deltaTime) / y)) +
                  (j1 / (y * y)) + j0 / y + velocityGoal * deltaTime + pos;
            velocity = eyedt * (j0 + j1 * deltaTime) + velocityGoal;
            acceleration = eyedt * (acceleration - j1 * y * deltaTime);
        }

        /// <summary>
        /// Given the current rotation, angular velocity and the desired rotation, returns the new
        /// rotation and angular velocity after deltaTime seconds.
        /// </summary>
        public static void SimpleSpringDamperImplicit(ref quaternion rot, ref float3 angularVel, quaternion rotGoal, float halfLife, float deltaTime)
        {
            var y = HalfLifeToDamping(halfLife) / 2.0f;
            var j0 = MathExtensions.QuaternionToScaledAngleAxis(MathExtensions.Abs(math.mul(rot, math.inverse(rotGoal))));
            var j1 = angularVel + j0 * y;
            var eyedt = FastNegativeExponent(y * deltaTime);

            rot = math.mul(MathExtensions.QuaternionFromScaledAngleAxis(eyedt * (j0 + j1 * deltaTime)), rotGoal);
            angularVel = eyedt * (angularVel - j1 * y * deltaTime);
        }
        /// <summary>
        /// Given the current position, velocity and the desired position, returns the new
        /// position and velocity after deltaTime seconds.
        /// </summary>
        public static void SimpleSpringDamperImplicit(ref float3 pos, ref float3 velocity, float3 posGoal, float halfLife, float deltaTime)
        {
            var y = HalfLifeToDamping(halfLife) / 2.0f;
            var j0 = pos - posGoal;
            var j1 = velocity + j0 * y;
            var eyedt = FastNegativeExponent(y * deltaTime);

            pos = eyedt * (j0 + j1 * deltaTime) + posGoal;
            velocity = eyedt * (velocity - j1 * y * deltaTime);
        }
        /// <summary>
        /// Given the current position, velocity and the desired position, returns the new
        /// position and velocity after deltaTime seconds.
        /// </summary>
        public static void SimpleSpringDamperImplicit(ref float2 pos, ref float2 velocity, float2 posGoal, float halfLife, float deltaTime)
        {
            var y = HalfLifeToDamping(halfLife) / 2.0f;
            var j0 = pos - posGoal;
            var j1 = velocity + j0 * y;
            var eyedt = FastNegativeExponent(y * deltaTime);

            pos = eyedt * (j0 + j1 * deltaTime) + posGoal;
            velocity = eyedt * (velocity - j1 * y * deltaTime);
        }

        /// <summary>
        /// Special type of SpringDamperImplicit when the desired rotation is the identity
        /// </summary>
        public static void DecaySpringDamperImplicit(ref quaternion rot, ref float3 angularVel, float halfLife, float deltaTime)
        {
            var y = HalfLifeToDamping(halfLife) / 2.0f;
            var j0 = MathExtensions.QuaternionToScaledAngleAxis(rot);
            var j1 = angularVel + j0 * y;
            var eyedt = FastNegativeExponent(y * deltaTime);

            rot = MathExtensions.QuaternionFromScaledAngleAxis(eyedt * (j0 + j1 * deltaTime));
            angularVel = eyedt * (angularVel - j1 * y * deltaTime);
        }
        /// <summary>
        /// Special type of SpringDamperImplicit when the desired position is 0
        /// </summary>
        public static void DecaySpringDamperImplicit(ref float3 pos, ref float3 velocity, float halfLife, float deltaTime)
        {
            var y = HalfLifeToDamping(halfLife) / 2.0f;
            var j1 = velocity + pos * y;
            var eyedt = FastNegativeExponent(y * deltaTime);

            pos = eyedt * (pos + j1 * deltaTime);
            velocity = eyedt * (velocity - j1 * y * deltaTime);
        }
        /// <summary>
        /// Special type of SpringDamperImplicit when the value is 0
        /// </summary>
        public static void DecaySpringDamperImplicit(ref float value, ref float velocity, float halfLife, float deltaTime)
        {
            var y = HalfLifeToDamping(halfLife) / 2.0f;
            var j1 = velocity + value * y;
            var eyedt = FastNegativeExponent(y * deltaTime);

            value = eyedt * (value + j1 * deltaTime);
            velocity = eyedt * (velocity - j1 * y * deltaTime);
        }

        private static float HalfLifeToDamping(float halfLife, float eps = 1e-5f)
        {
            const float ln2 = 0.69314718056f;
            return (4.0f * ln2) / (halfLife + eps);
        }

        private static float FastNegativeExponent(float x)
        {
            return 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);
        }
    }
}