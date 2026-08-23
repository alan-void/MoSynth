using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
    /// <summary>
    /// Something a crowd character should walk around: another character, or a static prop. Modelled
    /// as an upright cylinder — a circle on the ground plane plus a height band — which is all the
    /// avoidance maths needs and is cheap to intersect a ray against.
    /// </summary>
    /// <remarks>
    /// Registers with the scene's <see cref="ObstacleManager"/> while enabled. A non-static obstacle
    /// is expected to be a sibling of a crowd control input under a shared parent, which it finds at
    /// Awake; that link is what lets two characters agree which way to pass each other.
    /// </remarks>
    public class Obstacle : MonoBehaviour
    {
        public static readonly string EllipsesFeatureName = "FutureEllipse";

        /// <summary>Ground-plane radius of the cylinder, in metres.</summary>
        public float Radius = 1.0f;

        /// <summary>
        /// Set for props, which never move and never negotiate: static obstacles are skipped by
        /// steering entirely and always report their current position.
        /// </summary>
        public bool IsStatic = false;

        /// <summary>Vertical extent (min, max) relative to this transform, in metres.</summary>
        public float2 Height = new(0, 1);

        // The crowd control input this obstacle stands for, if any -- at most one of the two is set,
        // and both are null for a static obstacle. Used for steering coordination and predictions.
        private CrowdSplineControlInput _crowdSplineControlInput;
        private CrowdControlInput _crowdCharacter;

        private void Awake()
        {
            if (!IsStatic)
            {
                _crowdSplineControlInput = transform.parent.GetComponentInChildren<CrowdSplineControlInput>();
                _crowdCharacter = transform.parent.GetComponentInChildren<CrowdControlInput>();
            }

            Debug.Assert(IsStatic || (_crowdSplineControlInput != null || _crowdCharacter != null),
                         "Obstacle is not static but no CrowdSplineCharacterController or CrowdCharacterController is assigned. Please make sure a CharacterController is a component in any of the children of this component's parent transform.");
        }

        private void OnEnable()
        {
            ObstacleManager.Instance.RegisterObstacle(this);
        }

        private void OnDisable()
        {
            ObstacleManager.Instance.UnregisterObstacle(this);
        }

        /// <summary>
        /// This obstacle's own avoidance force, if it is a character. Read by whoever is avoiding it,
        /// so the two pick opposite sides. False for a static obstacle.
        /// </summary>
        public virtual bool GetCurrentSteering(out float2 steering)
        {
            if (_crowdSplineControlInput != null)
            {
                steering = _crowdSplineControlInput.Steering;
                return true;
            }
            else if (_crowdCharacter != null)
            {
                steering = _crowdCharacter.Steering;
                return true;
            }
            steering = float2.zero;
            return false;
        }

        /// <summary>
        /// Where this obstacle will be at a prediction horizon, on the ground plane — avoiding a
        /// moving character means aiming at where it is heading, not where it stands.
        /// </summary>
        /// <returns>Position; whether it is really a prediction; and the ellipse feature at that
        /// horizon, which is the swept shape a moving character occupies.</returns>
        /// <remarks>
        /// Only the static / <paramref name="forceCurrent"/> path works today — the prediction path
        /// needs the unimplemented feature read-back on
        /// <see cref="AnimationTools.MotionSynthesisComponent"/>.
        /// </remarks>
        public virtual (float3, bool, float4) GetProjWorldPosition(int trajectoryIndex, bool forceCurrent = false)
        {
            if (IsStatic || forceCurrent)
            {
                return (new float3(transform.position.x, 0.0f, transform.position.z), false, float4.zero);
            }
            else
            {
                if (_crowdSplineControlInput != null)
                {
                    return (_crowdSplineControlInput.motionSynthesizer.GetMainPositionFeature(trajectoryIndex),
                            true,
                            _crowdSplineControlInput.motionSynthesizer.GetEnvironmentFeature(EllipsesFeatureName, trajectoryIndex));
                }
                else
                {
                    return (_crowdCharacter.motionSynthesizer.GetMainPositionFeature(trajectoryIndex),
                            true,
                            _crowdCharacter.motionSynthesizer.GetEnvironmentFeature(EllipsesFeatureName, trajectoryIndex));
                }
            }
        }

        public float GetMinHeightWorld()
        {
            return transform.position.y + Height.x;
        }

        public float GetMaxHeightWorld()
        {
            return transform.position.y + Height.y;
        }

        /// <summary>
        /// Ray/circle intersection on the ground plane, used by steering. Reports entry and exit
        /// hits; rays pointing away from the obstacle miss.
        /// <a href="https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html">Method</a>.
        /// </summary>
        public bool Intersect(float2 rayOrigin, float2 rayDirection, out float2 hitPoint1, out float hitDistance1, out float2 hitPoint2, out float hitDistance2)
        {
            hitPoint1 = float2.zero;
            hitPoint2 = float2.zero;
            hitDistance1 = 0;
            hitDistance2 = 0;
            float2 center = new(transform.position.x, transform.position.z);
            float2 L = center - rayOrigin;
            float tca = math.dot(L, rayDirection);
            if (tca < 0) return false;
            float d2 = math.dot(L, L) - tca * tca;
            if (d2 > Radius * Radius) return false;
            float thc = math.sqrt(Radius * Radius - d2);
            float t0 = tca - thc;
            float t1 = tca + thc;
            if (t0 > t1)
            {
                (t1, t0) = (t0, t1);
            }
            if (t0 < 0)
            {
                t0 = t1; // If t0 is negative, let's use t1 instead.
                if (t0 < 0) return false; // Both t0 and t1 are negative.
            }
            hitPoint1 = rayOrigin + rayDirection * t0;
            hitDistance1 = t0;
            hitPoint2 = rayOrigin + rayDirection * t1;
            hitDistance2 = t1;
            return true;
        }

        protected virtual void OnDrawGizmos()
        {
            Gizmos.color = Color.red;
            Vector3 position = transform.position;
            GizmosExtensions.DrawWireCircle(new Vector3(position.x, GetMinHeightWorld(), position.z), Radius, Quaternion.identity);
            GizmosExtensions.DrawWireCircle(new Vector3(position.x, GetMaxHeightWorld(), position.z), Radius, Quaternion.identity);
            //Draw.Disc(new Vector3(position.x, GetMinHeightWorld(), position.z), Radius * MotionMatchingController.GIZMOS_MULTIPLIER, new Color(0.2f, 0.2f, 0.2f));
        }
    }
}