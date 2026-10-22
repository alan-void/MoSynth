using System;

namespace AnimationTools
{
/// <summary>
/// A causal gait phase, read tick by tick from contact flags: the runtime counterpart of
/// <see cref="GaitPhase"/>, for a stage that only sees the present and the past.
/// </summary>
/// <remarks>
/// Same conventions as <see cref="GaitPhase"/>: half a cycle per footfall, the cycle zeroed on a
/// right footfall, and a standing sweep by ground speed. Between footfalls the phase extrapolates
/// at the last stride's rate; at a footfall the error is closed by a decaying offset rather than a
/// snap, so the reported phase stays continuous. <c>Python/style/phase.py</c> ports this rule
/// line for line, and the golden vectors in <c>Python/tests/fixtures</c> hold the two together.
/// </remarks>
public sealed class OnlineGaitPhase
{
    /// <summary>Ground speed below which the character counts as standing, in m/s.</summary>
    public float standingSpeed = 0.1f;

    /// <summary>Seconds per cycle while standing.</summary>
    public float standingPeriod = 1f;

    /// <summary>Shortest gap between two footfalls of one foot; anything closer is contact chatter.</summary>
    public float minStepInterval = 0.15f;

    /// <summary>Half-life of the offset that hides a footfall's phase correction, in seconds.</summary>
    public float correctionHalfLife = 0.1f;

    /// <summary>Slowest stride rate a footfall may set, in radians per second.</summary>
    public float minRate = GaitPhase.Tau / 2f;

    /// <summary>Fastest stride rate a footfall may set, in radians per second.</summary>
    public float maxRate = GaitPhase.Tau / 0.4f;

    /// <summary>Seconds without a footfall before a slow character falls back to the standing sweep.</summary>
    public float standTimeout = 1f;

    private const int Left = 0;
    private const int Right = 1;
    private const int NoFoot = -1;

    // Double so the port and this agree to well inside the golden tolerance over long runs.
    private double _phase;
    private double _rate;
    private double _offset;
    private double _time;
    private double _lastFootfallTime;
    private int _lastFoot;
    private readonly double[] _lastFootTime = new double[2];
    private readonly bool[] _previousContact = new bool[2];
    private bool _initialized;

    public OnlineGaitPhase()
    {
        Reset();
    }

    /// <summary>The reported phase, in radians in <c>[0, Tau)</c>.</summary>
    public float Phase { get; private set; }

    /// <summary>The current stride rate, in radians per second.</summary>
    public float Rate { get; private set; }

    /// <summary>Forget everything; the next <see cref="Update"/> starts a fresh cycle.</summary>
    public void Reset()
    {
        _phase = 0.0;
        _rate = 0.0;
        _offset = 0.0;
        _time = 0.0;
        _lastFootfallTime = double.NegativeInfinity;
        _lastFoot = NoFoot;
        _lastFootTime[Left] = double.NegativeInfinity;
        _lastFootTime[Right] = double.NegativeInfinity;
        _previousContact[Left] = false;
        _previousContact[Right] = false;
        _initialized = false;
        Phase = 0f;
        Rate = 0f;
    }

    /// <summary>Advance one tick.</summary>
    /// <param name="dt">Seconds since the previous call.</param>
    /// <param name="leftContact">Whether the left foot is planted this tick.</param>
    /// <param name="rightContact">Whether the right foot is planted this tick.</param>
    /// <param name="groundSpeed">The character frame's planar speed, in m/s.</param>
    public void Update(float dt, bool leftContact, bool rightContact, float groundSpeed)
    {
        if (!_initialized)
        {
            // A foot already down is not a footfall: nothing says when it went down.
            _previousContact[Left] = leftContact;
            _previousContact[Right] = rightContact;
            _phase = 0.0;
            _rate = GaitPhase.Tau / (double)standingPeriod;
            _offset = 0.0;
            _time = 0.0;
            _initialized = true;
            Report();
            return;
        }

        _time += dt;
        _phase = Wrap(_phase + _rate * dt);

        DetectFootfall(Left, leftContact);
        DetectFootfall(Right, rightContact);
        _previousContact[Left] = leftContact;
        _previousContact[Right] = rightContact;

        if (groundSpeed < standingSpeed && _time - _lastFootfallTime >= standTimeout)
        {
            _rate = GaitPhase.Tau / (double)standingPeriod;
        }

        _offset = correctionHalfLife > 0f
            ? _offset * Math.Exp(-Math.Log(2.0) * dt / correctionHalfLife)
            : 0.0;

        Report();
    }

    private void DetectFootfall(int foot, bool contact)
    {
        if (!contact || _previousContact[foot]) return;
        if (_time - _lastFootTime[foot] < minStepInterval) return;

        var target = foot == Right ? 0.0 : Math.PI;

        if (!double.IsNegativeInfinity(_lastFootfallTime))
        {
            var interval = _time - _lastFootfallTime;
            // The same foot twice running is a missed contact, so a whole cycle went by.
            var cycles = foot == _lastFoot ? 1.0 : 0.5;
            if (interval > 0.0)
            {
                _rate = Math.Clamp(cycles * GaitPhase.Tau / interval, minRate, maxRate);
            }
        }

        var error = WrapToPi(target - _phase);
        _phase = Wrap(_phase + error);
        _offset -= error;

        _lastFootfallTime = _time;
        _lastFoot = foot;
        _lastFootTime[foot] = _time;
    }

    private void Report()
    {
        // A double just below Tau can round up to it as a float, leaving the half-open range.
        var phase = (float)Wrap(_phase + _offset);
        Phase = phase >= GaitPhase.Tau ? 0f : phase;
        Rate = (float)_rate;
    }

    /// <summary>Maps to <c>[0, Tau)</c>.</summary>
    private static double Wrap(double value)
    {
        const double tau = 2.0 * Math.PI;
        var wrapped = value - Math.Floor(value / tau) * tau;
        return wrapped >= tau ? 0.0 : wrapped;
    }

    /// <summary>Maps to <c>(-π, π]</c>.</summary>
    private static double WrapToPi(double value) => Math.PI - Wrap(Math.PI - value);
}
}
