using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Where one contact bone should be in the world this tick: pinned to the point it touched down at
/// while its contact holds, then eased back onto the animation once it lets go.
/// </summary>
/// <remarks>
/// The output is always the animated position plus a residual offset. Holding a lock makes that
/// offset whatever pins the bone; releasing hands it to a spring that decays it, so the output never
/// jumps. See <c>openwiki/animation-tools/root-following.md</c>.
/// </remarks>
public struct ContactLock
{
    private bool _wasInContact;
    private bool _isLocked;
    private float3 _lockPosition;
    private float3 _offset;
    private float3 _offsetVelocity;

    /// <summary>True while the bone is pinned to its lock point.</summary>
    public bool IsLocked => _isLocked;

    /// <summary>
    /// Advances one tick and returns where the bone should be.
    /// </summary>
    /// <param name="maxLockDistance">
    /// How far the animation may pull away from the lock point before the lock gives up rather than
    /// over-stretch the chain. 0 never gives up.
    /// </param>
    /// <param name="releaseHalfLife">Time to close half the gap back to the animation after release.</param>
    public float3 Step(bool contact, float3 animatedPosition, float maxLockDistance, float releaseHalfLife,
        float deltaTime)
    {
        // Latch where the bone is shown, not where it is animated, so touching down again while a
        // release is still easing out does not pop.
        if (contact && !_wasInContact)
        {
            _isLocked = true;
            _lockPosition = animatedPosition + _offset;
        }
        _wasInContact = contact;

        if (_isLocked)
        {
            var strayed = maxLockDistance > 0f && math.distance(animatedPosition, _lockPosition) > maxLockDistance;
            if (contact && !strayed)
            {
                _offset = _lockPosition - animatedPosition;
                return _lockPosition;
            }

            _isLocked = false;
            _offset = _lockPosition - animatedPosition;
            _offsetVelocity = float3.zero;
        }

        Spring.DecaySpringDamperImplicit(ref _offset, ref _offsetVelocity, releaseHalfLife, deltaTime);
        return animatedPosition + _offset;
    }

    /// <summary>Forgets any lock and residual, so the next tick follows the animation exactly.</summary>
    public void Reset() => this = default;
}
}
