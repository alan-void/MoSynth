using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// A stage that can show what it consumed on its last tick: the exact values it fed its search or
/// network, already taken out to world space, plus a few numbers for a readout.
/// </summary>
/// <remarks>
/// Read-only for the reader, and filled by the stage as part of its own <c>Apply</c>, so drawing it
/// cannot change the synthesis. A stage without it simply shows nothing.
/// </remarks>
public interface ISynthesisInputSource
{
    /// <summary>The stage's record of its last tick; never null once the stage has initialised.</summary>
    SynthesisInputs ConsumedInputs { get; }
}

/// <summary>Which side of the question a drawn input is on, which decides its colour.</summary>
public enum SynthesisInputRole
{
    /// <summary>What the method was asked for: a query, a trajectory window's future, a task.</summary>
    Request,

    /// <summary>What it answered with: the matched frame, the network's own prediction, the next state.</summary>
    Answer,

    /// <summary>Where the character already is: trajectory history, the current facing.</summary>
    State
}

/// <summary>One drawn input: a point, or an arrow from <see cref="Position"/> along <see cref="Vector"/>.</summary>
public struct SynthesisInputMarker
{
    public SynthesisInputRole Role;
    public bool IsArrow;

    /// <summary>World position of the point, or the arrow's tail.</summary>
    public float3 Position;

    /// <summary>World offset from tail to head, already scaled to a drawable length; zero for a point.</summary>
    public float3 Vector;
}

/// <summary>One labelled number, or labelled text, for a readout panel.</summary>
public struct SynthesisInputReadout
{
    public string Label;

    /// <summary>Shown instead of <see cref="Value"/> when set.</summary>
    public string Text;

    public float Value;

    /// <summary>A numeric format string for <see cref="Value"/>, e.g. <c>"0.00"</c>.</summary>
    public string Format;

    public string Unit;
}

/// <summary>
/// A reusable record of the inputs one stage consumed on its last tick. Capacities are fixed at
/// construction, so filling it every tick allocates nothing.
/// </summary>
/// <remarks>
/// Markers and readouts are cleared by the stage when it has something new to show, which need not
/// be every tick: a matcher's query only changes when it searches. The event flags describe only
/// the latest tick.
/// </remarks>
public sealed class SynthesisInputs
{
    private readonly SynthesisInputMarker[] _markers;
    private readonly SynthesisInputReadout[] _readouts;

    public SynthesisInputs(string title, int markerCapacity, int readoutCapacity)
    {
        Title = title;
        _markers = new SynthesisInputMarker[markerCapacity];
        _readouts = new SynthesisInputReadout[readoutCapacity];
    }

    /// <summary>What the inputs are, e.g. "matching query"; a constant, for a panel heading.</summary>
    public string Title { get; }

    public int MarkerCount { get; private set; }
    public int ReadoutCount { get; private set; }

    public ref readonly SynthesisInputMarker GetMarker(int index) => ref _markers[index];
    public ref readonly SynthesisInputReadout GetReadout(int index) => ref _readouts[index];

    /// <summary>What this method calls each role, for a legend; constants, e.g. "query" and "best match".</summary>
    public string RequestLabel { get; set; } = "request";
    public string AnswerLabel { get; set; } = "answer";
    public string StateLabel { get; set; } = "state";

    /// <summary>Whether the method searches at all, so a reader knows <see cref="Searched"/> means something.</summary>
    public bool Searches { get; set; }

    /// <summary>Whether the method searched on the latest tick.</summary>
    public bool Searched { get; set; }

    /// <summary>Whether the latest tick's search changed what the method plays.</summary>
    public bool AnswerChanged { get; set; }

    /// <summary>Whether <see cref="Phase"/> means anything for this method.</summary>
    public bool HasPhase { get; set; }

    /// <summary>The cycle phase the method was fed, as a fraction of a cycle in [0, 1).</summary>
    public float Phase { get; set; }

    /// <summary>Clears the per-tick event flags; call at the top of every tick.</summary>
    public void BeginTick()
    {
        Searched = false;
        AnswerChanged = false;
    }

    public void ClearMarkers() => MarkerCount = 0;
    public void ClearReadouts() => ReadoutCount = 0;

    public void AddPoint(SynthesisInputRole role, float3 position)
    {
        Add(new SynthesisInputMarker { Role = role, Position = position });
    }

    public void AddArrow(SynthesisInputRole role, float3 tail, float3 vector)
    {
        Add(new SynthesisInputMarker { Role = role, IsArrow = true, Position = tail, Vector = vector });
    }

    public void AddReadout(string label, float value, string format, string unit = null)
    {
        AddReadout(new SynthesisInputReadout { Label = label, Value = value, Format = format, Unit = unit });
    }

    public void AddReadout(string label, string text)
    {
        AddReadout(new SynthesisInputReadout { Label = label, Text = text });
    }

    // A full buffer drops the excess rather than growing, so a misjudged capacity costs a missing
    // marker instead of a per-tick allocation.
    private void Add(in SynthesisInputMarker marker)
    {
        if (MarkerCount == _markers.Length) return;
        _markers[MarkerCount++] = marker;
    }

    private void AddReadout(in SynthesisInputReadout readout)
    {
        if (ReadoutCount == _readouts.Length) return;
        _readouts[ReadoutCount++] = readout;
    }

    /// <summary>The first enabled stage of <paramref name="synthesizer"/> that records its inputs, or null.</summary>
    public static ISynthesisInputSource FindSource(MotionSynthesisComponent synthesizer)
    {
        if (synthesizer == null) return null;
        foreach (var stage in synthesizer.stages)
        {
            if (stage != null && stage.isEnabled && stage is ISynthesisInputSource source) return source;
        }

        return null;
    }
}
}
