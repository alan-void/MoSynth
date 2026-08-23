using System;
using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// One synthesis method in a sweep: a character prefab plus the knobs that make this entry
/// different from the others sharing that prefab.
/// </summary>
[Serializable]
public class BenchmarkMethod
{
    [Tooltip("Column name in the report. Must be unique and filename-safe.")]
    public string name = "Method";

    [Tooltip("Character to spawn. Needs a MotionSynthesisComponent and a control input implementing " +
             "IMotionSynthesisSplineControlInput somewhere in its hierarchy.")]
    public GameObject characterPrefab;

    [Tooltip("Applied to the spawned character before it wakes up. Lets one prefab cover several " +
             "methods (e.g. the four MotionField policies).")]
    [SerializeReference] [SubclassSelector]
    public List<BenchmarkOverride> overrides = new();

    /// <summary>Name plus the overrides, for logs and for the raw recording's filename.</summary>
    public string DescribeFull()
    {
        if (overrides == null || overrides.Count == 0) return name;

        var parts = new List<string>(overrides.Count);
        foreach (var o in overrides)
        {
            if (o != null) parts.Add(o.Describe());
        }

        return parts.Count == 0 ? name : $"{name} [{string.Join(", ", parts)}]";
    }
}

/// <summary>
/// The full definition of a benchmark sweep: which methods, over which paths, for how long, and
/// where the report goes. Everything a run needs, so a sweep is reproducible from this one asset.
/// </summary>
[CreateAssetMenu(fileName = "SynthesisBenchmark", menuName = "MoSynth/Synthesis Benchmark Config")]
public class SynthesisBenchmarkConfig : ScriptableObject
{
    [Header("What to sweep")]
    [Tooltip("Every method runs alone, once per path. Timings are contaminated if two characters " +
             "share a frame, which is why the sweep is sequential.")]
    public List<BenchmarkMethod> methods = new();

    [Tooltip("Folder of path prefabs, each with a SplineContainer at its root. Takes precedence " +
             "over the explicit list below when set. Resolved fresh at sweep start.")]
    public string pathPrefabFolder = "Assets/Benchmarks/Paths";

    [Tooltip("Explicit path prefabs, used only when the folder above is empty.")]
    public List<GameObject> pathPrefabs = new();

    [Header("Run length")]
    [Tooltip("Laps of the path to measure, counted after the settle time. Ignored for open splines, " +
             "which finish at the far end.")]
    [Min(0.1f)] public float lapsRequired = 1f;

    [Tooltip("Seconds at the start of each run excluded from every metric, while the character gets " +
             "up to speed from its spawn pose. Lap counting starts when this elapses.")]
    [Min(0f)] public float settleTime = 3f;

    [Tooltip("Hard limit per run, including the settle time. A run that hits it is reported with " +
             "timedOut set rather than being silently truncated.")]
    [Min(1f)] public float maxRunSeconds = 120f;

    [Header("Timing")]
    [Tooltip("Synthesis ticks per second for every run, unless a method overrides it.")]
    [Min(0f)] public float synthesisFrameRate = 30f;

    [Tooltip("Drive the run off a fixed timestep instead of the wall clock, so it fast-forwards as " +
             "quickly as the CPU allows and produces the same tick count every time. Turn off to " +
             "watch a sweep at normal speed in the Editor.")]
    public bool fixedTimestep = true;

    [Header("Metrics")]
    [Tooltip("Width in seconds of the moving average applied to finite-differenced speed before " +
             "computing velocity error. See PathFollowingMetricsCalculator.")]
    [Min(0f)] public float speedSmoothingWindow = 0.2f;

    [Tooltip("Also record the full pose buffer every tick. Not needed for any metric here; costs " +
             "roughly a kilobyte per tick and exists for offline analysis.")]
    public bool recordFullPose;

    [Header("Output")]
    [Tooltip("Where reports and raw recordings are written. Relative paths resolve against the " +
             "project root, next to Assets. Each sweep gets a timestamped subfolder.")]
    public string outputDirectory = "Benchmarks";

    /// <summary>
    /// Whether this config could actually run. Called from the inspector every repaint and from the
    /// CLI before a sweep starts, so it stays cheap and touches no assets.
    /// </summary>
    public bool TryValidate(out string error)
    {
        if (methods == null || methods.Count == 0)
        {
            error = "No methods configured.";
            return false;
        }

        var seen = new HashSet<string>();
        for (var i = 0; i < methods.Count; i++)
        {
            var method = methods[i];
            if (method == null)
            {
                error = $"Method {i} is null.";
                return false;
            }

            if (string.IsNullOrWhiteSpace(method.name))
            {
                error = $"Method {i} has no name.";
                return false;
            }

            if (!seen.Add(method.name))
            {
                error = $"Duplicate method name \"{method.name}\"; names become report rows and file names.";
                return false;
            }

            if (method.characterPrefab == null)
            {
                error = $"Method \"{method.name}\" has no character prefab.";
                return false;
            }
        }

        if (string.IsNullOrWhiteSpace(pathPrefabFolder) && (pathPrefabs == null || pathPrefabs.Count == 0))
        {
            error = "No path prefab folder and no explicit path prefabs.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            error = "No output directory.";
            return false;
        }

        error = null;
        return true;
    }
}
}
