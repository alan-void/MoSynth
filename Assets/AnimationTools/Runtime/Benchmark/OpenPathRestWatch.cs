namespace AnimationTools
{
/// <summary>
/// Decides when a character on an open path has come to rest short of its end: it has not advanced
/// along the path by at least a minimum step for a whole rest timeout.
/// </summary>
/// <remarks>
/// Progress is measured against the furthest point reached, so standing still, shuffling in place
/// and drifting backwards all count as rest, while a pause shorter than the timeout does not.
/// </remarks>
public sealed class OpenPathRestWatch
{
    private readonly float _restTimeout;
    private readonly float _minProgressT;

    private bool _hasSample;
    private float _furthestT;
    private float _secondsWithoutProgress;

    /// <param name="restTimeout">Seconds without forward progress before the character counts as at rest.</param>
    /// <param name="minProgressT">Smallest advance, as a normalized spline parameter, that counts as progress.</param>
    public OpenPathRestWatch(float restTimeout, float minProgressT)
    {
        _restTimeout = restTimeout;
        _minProgressT = minProgressT;
    }

    /// <summary>True once the character has gone a full rest timeout without forward progress.</summary>
    public bool HasRested => _hasSample && _secondsWithoutProgress >= _restTimeout;

    public void Sample(float t, float deltaTime)
    {
        if (!_hasSample || t >= _furthestT + _minProgressT)
        {
            _furthestT = t;
            _hasSample = true;
            _secondsWithoutProgress = 0f;
            return;
        }

        _secondsWithoutProgress += deltaTime;
    }
}
}
