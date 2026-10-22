using System;
using System.Collections.Generic;
using UnityEngine;
using Object = UnityEngine.Object;

namespace AnimationTools
{
/// <summary>How a benchmark video frames its run.</summary>
public enum BenchmarkVideoCamera
{
    /// <summary>Follows the character from behind and above, damped so root snaps never reach it.</summary>
    Chase,

    /// <summary>One fixed elevated view framing the whole path.</summary>
    Overview
}

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

    [Tooltip("Training seeds to run the character's learned stages (PFNN, LMM, the motion field's " +
             "optimal policy) with: one run per seed per path, each loading that seed's checkpoint. " +
             "Empty runs once, with each config's own seed. A method with no such stage must leave " +
             "this empty.")]
    public List<int> seeds = new();

    /// <summary>
    /// Whether the character, with <see cref="overrides"/> applied, runs an enabled stage that loads
    /// a per-seed checkpoint.
    /// </summary>
    /// <remarks>
    /// Overrides can decide it (a motion field run under the greedy policy loads none), so this
    /// instantiates the prefab inactive, applies them, and destroys it. Once per sweep, never per repaint.
    /// </remarks>
    public bool LoadsSeededCheckpoint()
    {
        if (characterPrefab == null) return false;

        var holder = new GameObject("SeedProbeHolder");
        holder.SetActive(false);
        try
        {
            var instance = Object.Instantiate(characterPrefab, holder.transform);
            if (overrides != null)
            {
                foreach (var methodOverride in overrides) methodOverride?.Apply(instance);
            }

            foreach (var synthesizer in instance.GetComponentsInChildren<MotionSynthesisComponent>(true))
            {
                foreach (var stage in synthesizer.stages)
                {
                    if (stage is { isEnabled: true } and ISeededStage { LoadsCheckpoint: true }) return true;
                }
            }

            return false;
        }
        finally
        {
            Object.DestroyImmediate(holder);
        }
    }

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
/// One entry of the list a sweep runs on every path: a method, and the training seed its learned
/// stages load, or none for the configs' own.
/// </summary>
public readonly struct BenchmarkMethodRun
{
    public readonly BenchmarkMethod Method;
    public readonly int? Seed;

    public BenchmarkMethodRun(BenchmarkMethod method, int? seed)
    {
        Method = method;
        Seed = seed;
    }

    /// <summary>The method's name, suffixed with the seed when there is one; names recordings and videos.</summary>
    public string Label => Seed.HasValue ? $"{Method.name}_s{Seed.Value}" : Method.name;
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

    [Tooltip("Thresholds that decide geometric foot contact for footskate, judged from the contact " +
             "bones' recorded positions rather than each method's own contact flags.")]
    public GroundContactSettings groundContact = new();

    [Tooltip("Hard limit per run, including the settle time. A run that hits it is reported with " +
             "timedOut set rather than being silently truncated.")]
    [Min(1f)] public float maxRunSeconds = 120f;

    [Tooltip("Open paths only: a run whose character makes no forward progress along the path for " +
             "this long ends with stoppedShort set and its remaining distance recorded, instead of " +
             "idling to the time limit. Keep it longer than any pause a method makes mid-path. " +
             "0 turns the rule off.")]
    [Min(0f)] public float restTimeout = 15f;

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

    [Tooltip("Also record the full pose buffer every tick, roughly a kilobyte per tick. None of " +
             "the in-Editor metrics read it; python -m benchmark.fmd needs it.")]
    public bool recordFullPose;

    [Header("Video")]
    [Tooltip("Also render every run to its own MP4 under videos/, with a JSON sidecar that " +
             "python -m benchmark.compose_videos reads to tile the methods side by side. Costs " +
             "meaningfully more time per sweep, and needs a graphics device: never -nographics.")]
    public bool recordVideo;

    [Tooltip("Chase follows the character from a damped 3/4 view; Overview holds one fixed view " +
             "framing the whole path.")]
    public BenchmarkVideoCamera videoCamera = BenchmarkVideoCamera.Chase;

    [Tooltip("Video resolution in pixels. Rounded down to even numbers, which H.264 requires.")]
    public int videoWidth = 1920;

    [Tooltip("Video resolution in pixels. Rounded down to even numbers, which H.264 requires.")]
    public int videoHeight = 1080;

    [Tooltip("While recording, add a ground grid, a shadowing light, the path, the steering target, " +
             "a root trail and footprints. Visual only: nothing it adds has a collider, so the " +
             "simulation and the metrics are unchanged.")]
    public bool videoSceneDressing = true;

    [Tooltip("While recording, draw the exact inputs the method consumed each tick, read back from the " +
             "stage rather than recomputed: the matching query against the matched frame, PFNN's " +
             "trajectory window and phase, the motion field's goal heading and speed. A HUD panel " +
             "lists the numbers. Visual only; the simulation and the metrics are unchanged.")]
    public bool videoShowMethodInputs = true;

    [Header("Output")]
    [Tooltip("Where reports and raw recordings are written. Relative paths resolve against the " +
             "project root, next to Assets. Each sweep gets a timestamped subfolder.")]
    public string outputDirectory = "Benchmarks";

    /// <summary>
    /// Every method once per seed, in method order and then seed order: what a sweep runs on each
    /// path.
    /// </summary>
    /// <param name="sweepSeeds">
    /// When set, the seeds of every method that <see cref="BenchmarkMethod.LoadsSeededCheckpoint"/>,
    /// replacing its own list; other methods run once. How a reproduction lists its seeds once.
    /// </param>
    public List<BenchmarkMethodRun> ExpandRuns(IReadOnlyList<int> sweepSeeds = null)
    {
        var runs = new List<BenchmarkMethodRun>();
        foreach (var method in methods)
        {
            IReadOnlyList<int> methodSeeds = method.seeds;
            if (sweepSeeds != null) methodSeeds = method.LoadsSeededCheckpoint() ? sweepSeeds : null;

            if (methodSeeds == null || methodSeeds.Count == 0)
            {
                runs.Add(new BenchmarkMethodRun(method, null));
                continue;
            }

            foreach (var seed in methodSeeds) runs.Add(new BenchmarkMethodRun(method, seed));
        }

        return runs;
    }

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

            if (method.seeds != null && new HashSet<int>(method.seeds).Count != method.seeds.Count)
            {
                error = $"Method \"{method.name}\" lists a seed twice; each would overwrite the other's recording.";
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
