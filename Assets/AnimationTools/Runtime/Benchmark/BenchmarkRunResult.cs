using System;

namespace AnimationTools
{
/// <summary>
/// One row of a sweep: what a single synthesis method did on a single path. Serializable so the
/// whole report round-trips through JsonUtility.
/// </summary>
[Serializable]
public class BenchmarkRunResult
{
    public string method;
    public string path;

    /// <summary>Sidecar-relative name of the .json manifest for this run's raw recording.</summary>
    public string recordingFile;

    /// <summary>True when the run hit its time limit before completing the required laps.</summary>
    public bool timedOut;

    /// <summary>Laps actually completed after the settle time. Below the required count means the run was cut short.</summary>
    public float completedLaps;

    public int framesTotal;
    public float durationSeconds;

    /// <summary>Set when the run failed outright (spawn error, recorder failure); the metric blocks are then unpopulated.</summary>
    public string error;

    public PathFollowingMetricsResult pathFollowing = new();
    public MotionQualityMetricsResult motionQuality = new();
    public SynthesisCostMetricsResult cost = new();
}
}
