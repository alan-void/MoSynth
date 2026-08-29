using Unity.Mathematics;

namespace MotionMatching
{
/// <summary>
/// Where a control input's simulation object has actually been: a fixed-size ring of timestamped
/// ground-plane samples, read back by interpolation anywhere inside the window it still holds.
/// </summary>
/// <remarks>
/// A trajectory feature with a negative prediction frame samples the database frames <em>before</em>
/// the query frame, so the query vector has to answer with the character's real past. A
/// spring-driven input cannot compute that — its spring says where the object is going, and running
/// it with a negative timestep does not run it backwards — so it has to remember instead. A path
/// follower does not need this: it reads its own path behind it.
/// <para>
/// Samples go in at the render rate and come out at the database rate, so reads interpolate. A
/// request older than the oldest retained sample fails rather than clamping, so a caller cannot
/// mistake the start of a run for a long stand still.
/// </para>
/// </remarks>
public sealed class TrajectoryHistory
{
    private struct Sample
    {
        public float Time;
        public float2 Position;
        public float2 Forward;
    }

    private readonly Sample[] _samples;

    /// <summary>Index the next <see cref="Record"/> writes to.</summary>
    private int _head;

    private int _count;

    /// <summary>How many samples fit before the oldest starts being dropped.</summary>
    public int Capacity => _samples.Length;

    /// <summary>How many samples are currently held, up to <see cref="Capacity"/>.</summary>
    public int Count => _count;

    /// <summary>Timestamp of the oldest retained sample. Undefined while <see cref="Count"/> is 0.</summary>
    public float OldestTime => At(0).Time;

    /// <summary>Timestamp of the newest sample. Undefined while <see cref="Count"/> is 0.</summary>
    public float NewestTime => At(_count - 1).Time;

    public TrajectoryHistory(int capacity)
    {
        _samples = new Sample[math.max(2, capacity)];
    }

    /// <summary>
    /// Forgets everything recorded so far. Call this whenever the simulation object moves without
    /// travelling — a teleport, a respawn — since interpolating across that jump would report a
    /// path the character never took.
    /// </summary>
    public void Clear()
    {
        _head = 0;
        _count = 0;
    }

    /// <summary>
    /// Appends a sample, dropping the oldest once <see cref="Capacity"/> is reached. A timestamp at
    /// or before the newest one overwrites that sample instead of appending, which keeps the buffer
    /// strictly increasing in time so <see cref="TrySample"/> can bracket a query.
    /// </summary>
    public void Record(float time, float2 position, float2 forward)
    {
        var sample = new Sample { Time = time, Position = position, Forward = forward };

        if (_count > 0 && time <= NewestTime)
        {
            _samples[(_head - 1 + _samples.Length) % _samples.Length] = sample;
            return;
        }

        _samples[_head] = sample;
        _head = (_head + 1) % _samples.Length;
        if (_count < _samples.Length) _count++;
    }

    /// <summary>
    /// The recorded state at <paramref name="time"/>, interpolated between the two samples that
    /// bracket it. A time at or after the newest sample reports that sample; a time before the
    /// oldest one returns false, because the buffer genuinely does not know.
    /// </summary>
    public bool TrySample(float time, out float2 position, out float2 forward)
    {
        position = default;
        forward = default;
        if (_count == 0) return false;

        if (time >= NewestTime)
        {
            var newest = At(_count - 1);
            position = newest.Position;
            forward = newest.Forward;
            return true;
        }

        if (time < OldestTime) return false;

        // Lowest ordinal whose timestamp is at or after the query; there is one, since the query
        // sits inside [OldestTime, NewestTime).
        var low = 0;
        var high = _count - 1;
        while (low < high)
        {
            var middle = (low + high) / 2;
            if (At(middle).Time < time) low = middle + 1;
            else high = middle;
        }

        var after = At(low);
        if (low == 0)
        {
            position = after.Position;
            forward = after.Forward;
            return true;
        }

        var before = At(low - 1);
        var span = after.Time - before.Time;
        var t = span > 0f ? (time - before.Time) / span : 0f;

        position = math.lerp(before.Position, after.Position, t);
        forward = math.normalizesafe(math.lerp(before.Forward, after.Forward, t), after.Forward);
        return true;
    }

    /// <summary>The <paramref name="ordinal"/>-th oldest retained sample.</summary>
    private Sample At(int ordinal) => _samples[(_head - _count + ordinal + _samples.Length) % _samples.Length];
}
}
