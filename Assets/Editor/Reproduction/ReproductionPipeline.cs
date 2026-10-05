using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AnimationTools;
using AnimationTools.Editor;
using Lmm;
using Lmm.Editor;
using MotionField;
using MotionField.Editor;
using MotionMatching;
using Pfnn;
using Pfnn.Editor;
using UnityEditor;
using UnityEngine;

namespace Reproduction
{
/// <summary>
/// Validates, builds and trains every config a <see cref="ReproductionManifest"/>'s benchmarks use,
/// one step per entry point: <c>-executeMethod Reproduction.ReproductionPipeline.Validate</c>,
/// <c>.BuildDatabases</c> or <c>.Train</c>. The sweeps themselves run through
/// <see cref="SynthesisBenchmarkCli.Run"/>.
/// </summary>
/// <remarks>
/// Each step is synchronous, because a batchmode Editor does not reliably pump
/// <see cref="EditorApplication.delayCall"/>, and in batchmode exits with 0 when everything
/// succeeded and 1 otherwise. One config failing never stops the others. Arguments:
/// <c>-reproManifest &lt;asset path&gt;</c>, <c>-reproLog &lt;folder&gt;</c> (relative to the project
/// root), and <c>-reproOnly &lt;name,name&gt;</c> to restrict every step to those config assets.
/// </remarks>
public static class ReproductionPipeline
{
    private const string ManifestArgument = "-reproManifest";
    private const string LogArgument = "-reproLog";
    private const string OnlyArgument = "-reproOnly";

    private const string DefaultManifestPath = "Assets/Benchmarks/PaperReproduction.asset";
    private const string DefaultLogDirectory = "Benchmarks/reproduction";

    private const string LogPrefix = "[Reproduction]";

    [MenuItem("MoSynth/Reproduction/1 Validate")]
    public static void Validate() => RunStep("Validate", ValidateAll);

    [MenuItem("MoSynth/Reproduction/2 Build Databases")]
    public static void BuildDatabases() => RunStep("Build Databases", BuildAll);

    [MenuItem("MoSynth/Reproduction/3 Train")]
    public static void Train() => RunStep("Train", TrainAll);

    /// <summary>What one invocation works on, resolved from the command line.</summary>
    private sealed class Invocation
    {
        public ReproductionManifest Manifest;
        public string ManifestPath;
        public string LogDirectory;
        public List<ScriptableObject> Configs;
    }

    private static void RunStep(string stepName, Func<Invocation, bool> step)
    {
        var ok = false;
        try
        {
            var invocation = ResolveInvocation();
            if (invocation != null)
            {
                Debug.Log($"{LogPrefix} {stepName}: {invocation.Configs.Count} config(s) from " +
                          $"\"{invocation.ManifestPath}\": " +
                          string.Join(", ", invocation.Configs.Select(c => c.name)));
                ok = step(invocation);
            }
        }
        catch (Exception e)
        {
            Debug.LogException(e);
        }

        if (ok) Debug.Log($"{LogPrefix} {stepName} succeeded.");
        else Debug.LogError($"{LogPrefix} {stepName} failed (see the errors above).");

        if (Application.isBatchMode) EditorApplication.Exit(ok ? 0 : 1);
    }

    private static Invocation ResolveInvocation()
    {
        var manifestPath = SynthesisBenchmarkCli.GetArgument(ManifestArgument);
        if (string.IsNullOrEmpty(manifestPath)) manifestPath = DefaultManifestPath;

        var manifest = AssetDatabase.LoadAssetAtPath<ReproductionManifest>(manifestPath);
        if (manifest == null)
        {
            Debug.LogError($"{LogPrefix} No ReproductionManifest at \"{manifestPath}\".");
            return null;
        }

        var logDirectory = SynthesisBenchmarkCli.GetArgument(LogArgument);
        if (string.IsNullOrEmpty(logDirectory)) logDirectory = DefaultLogDirectory;

        return new Invocation
        {
            Manifest = manifest,
            ManifestPath = manifestPath,
            LogDirectory = SynthesisBenchmarkLauncher.ResolveOutputDirectory(logDirectory),
            Configs = FilterByName(DeriveConfigs(manifest), SynthesisBenchmarkCli.GetArgument(OnlyArgument)),
        };
    }

    // --- Config derivation ------------------------------------------------------------------

    /// <summary>
    /// Every config asset the manifest's method prefabs run, deduplicated, in first-seen order.
    /// </summary>
    /// <remarks>
    /// No <see cref="BenchmarkOverride"/> swaps a config, so the prefab alone decides which ones a
    /// method uses.
    /// </remarks>
    private static List<ScriptableObject> DeriveConfigs(ReproductionManifest manifest)
    {
        var configs = new List<ScriptableObject>();
        foreach (var benchmark in manifest.benchmarks)
        {
            if (benchmark == null) continue;
            foreach (var method in benchmark.methods)
            {
                if (method?.characterPrefab == null) continue;
                foreach (var synthesizer in method.characterPrefab.GetComponentsInChildren<MotionSynthesisComponent>(true))
                {
                    foreach (var stage in synthesizer.stages) AddStageConfigs(stage, configs);
                }
            }
        }

        return configs;
    }

    private static void AddStageConfigs(MoSynthStage stage, List<ScriptableObject> configs)
    {
        switch (stage)
        {
            case MotionMatchingStage motionMatching:
                AddDistinct(configs, motionMatching.mmData);
                break;
            case PoseSetVisualizerStage visualizer:
                AddDistinct(configs, visualizer.mmData);
                break;
            case MotionFieldStage motionField:
                AddDistinct(configs, motionField.config);
                break;
            case PfnnStage pfnn:
                AddDistinct(configs, pfnn.config);
                break;
            case LmmStage lmm:
                // LMM learns its MotionMatchingData's database, so that has to be built too.
                if (lmm.config != null) AddDistinct(configs, lmm.config.mmData);
                AddDistinct(configs, lmm.config);
                break;
        }
    }

    private static void AddDistinct(List<ScriptableObject> configs, ScriptableObject config)
    {
        if (config != null && !configs.Contains(config)) configs.Add(config);
    }

    private static List<ScriptableObject> FilterByName(List<ScriptableObject> configs, string commaSeparatedNames)
    {
        if (string.IsNullOrWhiteSpace(commaSeparatedNames)) return configs;

        var names = new HashSet<string>(
            commaSeparatedNames.Split(',').Select(n => n.Trim()).Where(n => n.Length > 0),
            StringComparer.OrdinalIgnoreCase);

        foreach (var name in names.Where(n => configs.All(c => !string.Equals(c.name, n, StringComparison.OrdinalIgnoreCase))))
        {
            Debug.LogWarning($"{LogPrefix} {OnlyArgument} names \"{name}\", which no benchmark method uses.");
        }

        return configs.Where(c => names.Contains(c.name)).ToList();
    }

    // --- Validate ---------------------------------------------------------------------------

    [Serializable]
    private class ValidationEntry
    {
        public string asset;
        public string path;
        public string kind;
        public bool ok;
        public List<string> errors = new();
    }

    [Serializable]
    private class ValidationLog
    {
        public string startedUtc;
        public string manifest;
        public bool ok;
        public List<ValidationEntry> entries = new();
    }

    private static bool ValidateAll(Invocation invocation)
    {
        var log = new ValidationLog { startedUtc = UtcNow(), manifest = invocation.ManifestPath };

        var manifestEntry = NewValidationEntry(invocation.Manifest);
        if (invocation.Manifest.benchmarks.Count == 0) manifestEntry.errors.Add("Lists no benchmarks.");
        if (invocation.Configs.Count == 0) manifestEntry.errors.Add("Its benchmarks use no config to build or train.");
        CollectMissingReferences(invocation.Manifest, invocation.Manifest.name, manifestEntry.errors);
        log.entries.Add(manifestEntry);

        for (var i = 0; i < invocation.Manifest.benchmarks.Count; i++)
        {
            var benchmark = invocation.Manifest.benchmarks[i];
            if (benchmark == null)
            {
                manifestEntry.errors.Add($"Benchmark {i} is unassigned or missing.");
                continue;
            }

            var entry = NewValidationEntry(benchmark);
            if (!benchmark.TryValidate(out var error)) entry.errors.Add(error);
            CollectMissingReferences(benchmark, benchmark.name, entry.errors);
            foreach (var method in benchmark.methods)
            {
                if (method?.characterPrefab != null)
                    CollectPrefabProblems(method.characterPrefab, $"method \"{method.name}\"", entry.errors);
            }

            log.entries.Add(entry);
        }

        foreach (var config in invocation.Configs)
        {
            var entry = NewValidationEntry(config);
            if (!TryValidateConfig(config, out var error)) entry.errors.Add(error);
            CollectMissingReferences(config, config.name, entry.errors);
            log.entries.Add(entry);
        }

        foreach (var entry in log.entries) entry.ok = entry.errors.Count == 0;
        log.ok = log.entries.All(e => e.ok);

        var report = new StringBuilder($"{LogPrefix} Validation of \"{invocation.ManifestPath}\":\n");
        foreach (var entry in log.entries)
        {
            report.AppendLine($"  {(entry.ok ? "OK  " : "FAIL")} {entry.kind} {entry.asset}");
            foreach (var error in entry.errors) report.AppendLine($"         {error}");
        }

        if (log.ok) Debug.Log(report.ToString());
        else Debug.LogError(report.ToString());

        WriteLog(invocation, "validate.json", log);
        return log.ok;
    }

    private static ValidationEntry NewValidationEntry(UnityEngine.Object asset) => new()
    {
        asset = asset.name,
        path = AssetDatabase.GetAssetPath(asset),
        kind = asset.GetType().Name,
    };

    private static bool TryValidateConfig(ScriptableObject config, out string error)
    {
        switch (config)
        {
            case MotionMatchingData motionMatching: return motionMatching.TryValidate(out error);
            case MotionFieldConfig motionField: return motionField.TryValidate(out error);
            case PfnnConfig pfnn: return pfnn.TryValidate(out error);
            case LmmConfig lmm: return lmm.TryValidate(out error);
            default:
                error = null;
                return true;
        }
    }

    /// <summary>Missing scripts and dangling references anywhere in a prefab's hierarchy.</summary>
    private static void CollectPrefabProblems(GameObject prefab, string label, List<string> errors)
    {
        foreach (var transform in prefab.GetComponentsInChildren<Transform>(true))
        {
            var gameObject = transform.gameObject;
            var missingScripts = GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(gameObject);
            if (missingScripts > 0)
                errors.Add($"{label}: \"{prefab.name}/{gameObject.name}\" has {missingScripts} missing script(s).");

            foreach (var component in gameObject.GetComponents<Component>())
            {
                if (component != null)
                    CollectMissingReferences(component, $"{label}: {prefab.name}/{gameObject.name}.{component.GetType().Name}", errors);
            }
        }
    }

    /// <summary>
    /// Object references that were assigned but whose target no longer exists — what the Inspector
    /// draws as "Missing", as opposed to a field left at None.
    /// </summary>
    private static void CollectMissingReferences(UnityEngine.Object target, string label, List<string> errors)
    {
        using var serializedObject = new SerializedObject(target);
        var property = serializedObject.GetIterator();
        while (property.Next(true))
        {
            if (property.propertyType != SerializedPropertyType.ObjectReference) continue;
            if (property.objectReferenceValue == null && property.objectReferenceEntityIdValue != EntityId.None)
                errors.Add($"{label}: missing reference at {property.propertyPath}.");
        }
    }

    // --- Build databases --------------------------------------------------------------------

    [Serializable]
    private class BuildEntry
    {
        public string config;
        public string path;
        public string kind;
        public bool ok;
        public double seconds;
        public string error;
    }

    [Serializable]
    private class BuildLog
    {
        public string startedUtc;
        public string manifest;
        public MachineRecord machine;
        public bool ok = true;
        public List<BuildEntry> entries = new();
    }

    private static bool BuildAll(Invocation invocation)
    {
        var log = new BuildLog
        {
            startedUtc = UtcNow(),
            manifest = invocation.ManifestPath,
            machine = MachineRecord.Current(),
        };
        var fileName = $"build_log_{SynthesisBenchmarkLauncher.TimestampFolderName()}.json";

        foreach (var config in invocation.Configs)
        {
            if (config is LmmConfig lmm)
            {
                Debug.Log($"{LogPrefix} {lmm.name} owns no database; it learns " +
                          $"\"{(lmm.mmData != null ? lmm.mmData.name : "nothing")}\"'s.");
                continue;
            }

            var entry = new BuildEntry
            {
                config = config.name,
                path = AssetDatabase.GetAssetPath(config),
                kind = config.GetType().Name,
            };

            Debug.Log($"{LogPrefix} BEGIN build {entry.kind} {config.name}");
            var watch = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                entry.error = BuildDatabase(config);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
                entry.error = e.Message;
            }

            entry.seconds = watch.Elapsed.TotalSeconds;
            entry.ok = entry.error == null;
            log.ok &= entry.ok;
            log.entries.Add(entry);
            LogFinished("build", entry.kind, config.name, entry.ok, entry.seconds, entry.error);

            // Rewritten after every config, so a crash late in a long run keeps what finished.
            WriteLog(invocation, fileName, log);
        }

        WriteLog(invocation, fileName, log);
        return log.ok;
    }

    /// <summary>Builds one config's database. Returns null on success, else why it failed.</summary>
    private static string BuildDatabase(ScriptableObject config)
    {
        const string notWritten = "The database was not written; see the console.";
        switch (config)
        {
            case MotionMatchingData motionMatching:
                // GenerateDatabases logs and returns on an invalid asset rather than reporting it.
                if (!motionMatching.TryValidate(out var error)) return error;
                MotionMatchingDataEditor.GenerateDatabases(motionMatching);
                return null;
            case PfnnConfig pfnn:
                return PfnnConfigEditor.GeneratePoseDatabase(pfnn) ? null : notWritten;
            case MotionFieldConfig motionField:
                return MotionFieldConfigEditor.GeneratePoseDatabase(motionField) ? null : notWritten;
            default:
                return $"No database build step for {config.GetType().Name}.";
        }
    }

    // --- Train ------------------------------------------------------------------------------

    [Serializable]
    private class TrainingRun
    {
        public string config;
        public string path;
        public string method;
        public string startedUtc;
        public double seconds;
        public bool ok;
        public bool seeded;
        public int seed;

        // The Python trainer's own summary, as JSON text: its contents differ per method.
        public string summaryJson;
        public string error;
    }

    [Serializable]
    private class TrainingLog
    {
        public string startedUtc;
        public string manifest;
        public MachineRecord machine;
        public bool ok = true;
        public List<TrainingRun> runs = new();
    }

    private static bool TrainAll(Invocation invocation)
    {
        var log = new TrainingLog
        {
            startedUtc = UtcNow(),
            manifest = invocation.ManifestPath,
            machine = MachineRecord.Current(),
        };
        var fileName = $"training_log_{SynthesisBenchmarkLauncher.TimestampFolderName()}.json";

        void Record(TrainingRun run)
        {
            log.ok &= run.ok;
            log.runs.Add(run);
            WriteLog(invocation, fileName, log);
        }

        foreach (var config in invocation.Configs)
        {
            switch (config)
            {
                case LmmConfig lmm:
                    Record(TrainOne(lmm, "Lmm", lmm.seed, () => LmmTraining.Run(lmm)));
                    break;
                case PfnnConfig pfnn:
                    Record(TrainOne(pfnn, "Pfnn", pfnn.seed, () => PfnnTraining.Run(pfnn)));
                    break;
                case MotionFieldConfig motionField:
                    // Value iteration draws no random numbers, so it takes no seed.
                    Record(TrainOne(motionField, "MotionField", null,
                        () => MotionFieldConfigEditor.TrainMotionField(motionField)));
                    // The visualizer's embedding indexes the pose database, so it goes stale with it.
                    Record(TrainOne(motionField, "MotionFieldEmbedding", motionField.umapSeed,
                        () => MotionFieldConfigEditor.ComputeEmbedding(motionField)));
                    break;
            }
        }

        WriteLog(invocation, fileName, log);
        return log.ok;
    }

    /// <param name="train">Runs the training; returns its summary as JSON, or null when it failed.</param>
    private static TrainingRun TrainOne(ScriptableObject config, string method, int? seed, Func<string> train)
    {
        var run = new TrainingRun
        {
            config = config.name,
            path = AssetDatabase.GetAssetPath(config),
            method = method,
            startedUtc = UtcNow(),
            seeded = seed.HasValue,
            seed = seed ?? 0,
        };

        Debug.Log($"{LogPrefix} BEGIN train {method} {config.name}");
        var watch = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            run.summaryJson = train();
            if (run.summaryJson == null) run.error = "Training failed; see the console.";
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            run.error = e.Message;
        }

        run.seconds = watch.Elapsed.TotalSeconds;
        run.ok = run.error == null;
        LogFinished("train", method, config.name, run.ok, run.seconds, run.error);
        return run;
    }

    // --- Shared -----------------------------------------------------------------------------

    [Serializable]
    private class MachineRecord
    {
        public string processorType;
        public int processorCount;
        public string graphicsDeviceName;
        public int systemMemoryMb;
        public string operatingSystem;
        public string unityVersion;
        public bool batchMode;

        public static MachineRecord Current() => new()
        {
            processorType = SystemInfo.processorType,
            processorCount = SystemInfo.processorCount,
            graphicsDeviceName = SystemInfo.graphicsDeviceName,
            systemMemoryMb = SystemInfo.systemMemorySize,
            operatingSystem = SystemInfo.operatingSystem,
            unityVersion = Application.unityVersion,
            batchMode = Application.isBatchMode,
        };
    }

    private static string UtcNow() => DateTime.UtcNow.ToString("o");

    private static void LogFinished(string verb, string kind, string name, bool ok, double seconds, string error)
    {
        if (ok) Debug.Log($"{LogPrefix} END {verb} {kind} {name} ({seconds / 60:0.0} min)");
        else Debug.LogError($"{LogPrefix} FAIL {verb} {kind} {name} ({seconds / 60:0.0} min): {error}");
    }

    private static void WriteLog(Invocation invocation, string fileName, object log)
    {
        Directory.CreateDirectory(invocation.LogDirectory);
        var path = Path.Combine(invocation.LogDirectory, fileName);
        File.WriteAllText(path, JsonUtility.ToJson(log, true));
        Debug.Log($"{LogPrefix} Wrote {path}");
    }
}
}
