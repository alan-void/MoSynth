using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Turns a completed sweep's <see cref="BenchmarkRunResult"/> list into the files a human or a
/// script actually reads: a flat CSV for spreadsheets, a full JSON dump for anything the CSV
/// drops (per-stage cost breakdown), and a console table for a quick look right after the run.
/// </summary>
public static class BenchmarkReportWriter
{
    private static readonly string[] CsvHeader =
    {
        "method", "path", "seed", "completedLaps", "timedOut", "stoppedShort", "remainingDistance",
        "framesTotal", "durationSeconds",
        "framesEvaluated", "meanTrajectoryError", "meanHeadingErrorDeg", "maxHeadingErrorDeg",
        "meanActualSpeed", "targetSpeed", "meanVelocityError", "footskatePerMeter",
        "meanFootskateSpeed", "contactFraction", "flagFootskatePerMeter", "flagMeanFootskateSpeed",
        "flagOnsetHeightError", "flagContactFraction", "contactFlagAgreement", "rootJerkMean", "rootJerkP95",
        "rootVerticalJerkMean", "rootVerticalJerkP95", "discontinuitiesPerSecond", "applyMsMean", "applyMsP50", "applyMsP95", "applyMsMax",
        "gcBytesPerTick", "recordingFile", "error"
    };

    /// <summary>The results.csv metrics averaged per method in averages.csv, in column order.</summary>
    private static readonly (string column, Func<BenchmarkRunResult, float> value)[] AveragedMetrics =
    {
        ("completedLaps", r => r.completedLaps),
        ("remainingDistance", r => r.remainingDistance),
        ("durationSeconds", r => r.durationSeconds),
        ("meanTrajectoryError", r => r.pathFollowing.meanTrajectoryError),
        ("meanHeadingErrorDeg", r => r.pathFollowing.meanHeadingErrorDeg),
        ("maxHeadingErrorDeg", r => r.pathFollowing.maxHeadingErrorDeg),
        ("meanActualSpeed", r => r.pathFollowing.meanActualSpeed),
        ("targetSpeed", r => r.pathFollowing.targetSpeed),
        ("meanVelocityError", r => r.pathFollowing.meanVelocityError),
        ("footskatePerMeter", r => r.motionQuality.footskatePerMeter),
        ("meanFootskateSpeed", r => r.motionQuality.meanFootskateSpeed),
        ("contactFraction", r => r.motionQuality.contactFraction),
        ("flagFootskatePerMeter", r => r.motionQuality.flagFootskatePerMeter),
        ("flagMeanFootskateSpeed", r => r.motionQuality.flagMeanFootskateSpeed),
        ("flagOnsetHeightError", r => r.motionQuality.flagOnsetHeightError),
        ("flagContactFraction", r => r.motionQuality.flagContactFraction),
        ("contactFlagAgreement", r => r.motionQuality.contactFlagAgreement),
        ("rootJerkMean", r => r.motionQuality.rootJerkMean),
        ("rootJerkP95", r => r.motionQuality.rootJerkP95),
        ("rootVerticalJerkMean", r => r.motionQuality.rootVerticalJerkMean),
        ("rootVerticalJerkP95", r => r.motionQuality.rootVerticalJerkP95),
        ("discontinuitiesPerSecond", r => r.motionQuality.discontinuitiesPerSecond),
        ("applyMsMean", r => r.cost.applyMsMean),
        ("applyMsP50", r => r.cost.applyMsP50),
        ("applyMsP95", r => r.cost.applyMsP95),
        ("applyMsMax", r => r.cost.applyMsMax),
        ("gcBytesPerTick", r => r.cost.gcBytesPerTick)
    };

    /// <summary>
    /// Writes results.csv, results.json and averages.csv into <paramref name="outputDirectory"/>
    /// and logs a summary table. Returns the absolute path of results.csv.
    /// </summary>
    public static string Write(
        string outputDirectory,
        SynthesisBenchmarkConfig config,
        string configAssetPath,
        IReadOnlyList<BenchmarkRunResult> results)
    {
        var csvPath = Path.Combine(outputDirectory, "results.csv");
        var jsonPath = Path.Combine(outputDirectory, "results.json");

        WriteCsv(csvPath, results);
        WriteJson(jsonPath, config, configAssetPath, results);
        WriteAverages(Path.Combine(outputDirectory, "averages.csv"), results);
        LogSummary(results);

        return csvPath;
    }

    /// <summary>
    /// One row per method, in first-run order, with each metric's mean over that method's runs,
    /// every seed and path weighing the same. Timed-out runs are averaged in and counted, errored
    /// runs are only counted because their metrics were never computed, and NaN values
    /// (remainingDistance on a closed path) are skipped.
    /// </summary>
    /// <remarks>
    /// Each metric's <c>SeedStd</c> column is the sample standard deviation of the per-seed means,
    /// the spread a reader needs to tell a method from a lucky draw; NaN below two seeds.
    /// <c>benchmark.report</c> turns the same per-seed means into a confidence interval.
    /// </remarks>
    private static void WriteAverages(string path, IReadOnlyList<BenchmarkRunResult> results)
    {
        var byMethod = new Dictionary<string, List<BenchmarkRunResult>>();
        var methodOrder = new List<string>();
        foreach (var result in results)
        {
            if (!byMethod.TryGetValue(result.method, out var runs))
            {
                runs = new List<BenchmarkRunResult>();
                byMethod.Add(result.method, runs);
                methodOrder.Add(result.method);
            }
            runs.Add(result);
        }

        var header = new List<string> { "method", "runs", "seeds", "timedOut", "stoppedShort", "errors" };
        foreach (var metric in AveragedMetrics)
        {
            header.Add(metric.column);
            header.Add(metric.column + "SeedStd");
        }

        var sb = new StringBuilder();
        AppendRow(sb, header.ToArray());

        var row = new string[header.Count];
        foreach (var method in methodOrder)
        {
            var runs = byMethod[method];
            var measured = runs.FindAll(r => string.IsNullOrEmpty(r.error));
            var bySeed = GroupBySeed(measured);

            var column = 0;
            row[column++] = method;
            row[column++] = runs.Count.ToString(CultureInfo.InvariantCulture);
            row[column++] = CountSeeds(runs).ToString(CultureInfo.InvariantCulture);
            row[column++] = runs.FindAll(r => r.timedOut).Count.ToString(CultureInfo.InvariantCulture);
            row[column++] = runs.FindAll(r => r.stoppedShort).Count.ToString(CultureInfo.InvariantCulture);
            row[column++] = (runs.Count - measured.Count).ToString(CultureInfo.InvariantCulture);
            foreach (var metric in AveragedMetrics)
            {
                row[column++] = FormatFloat(MeanIgnoringNaN(measured, metric.value));
                row[column++] = FormatFloat(SeedStandardDeviation(bySeed, metric.value));
            }

            AppendRow(sb, row);
        }

        File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
    }

    /// <summary>Runs grouped by training seed; every unseeded run falls in one group of its own.</summary>
    private static List<List<BenchmarkRunResult>> GroupBySeed(List<BenchmarkRunResult> runs)
    {
        var groups = new Dictionary<(bool seeded, int seed), List<BenchmarkRunResult>>();
        var order = new List<List<BenchmarkRunResult>>();
        foreach (var run in runs)
        {
            var key = (run.seeded, run.seeded ? run.seed : 0);
            if (!groups.TryGetValue(key, out var group))
            {
                group = new List<BenchmarkRunResult>();
                groups.Add(key, group);
                order.Add(group);
            }
            group.Add(run);
        }
        return order;
    }

    private static int CountSeeds(List<BenchmarkRunResult> runs)
    {
        var seeds = new HashSet<int>();
        foreach (var run in runs)
        {
            if (run.seeded) seeds.Add(run.seed);
        }
        return seeds.Count;
    }

    /// <summary>Sample standard deviation of the per-seed means, or NaN when fewer than two seeds have one.</summary>
    private static float SeedStandardDeviation(List<List<BenchmarkRunResult>> bySeed,
        Func<BenchmarkRunResult, float> value)
    {
        var means = new List<double>(bySeed.Count);
        foreach (var group in bySeed)
        {
            var mean = MeanIgnoringNaN(group, value);
            if (!float.IsNaN(mean)) means.Add(mean);
        }

        if (means.Count < 2) return float.NaN;

        var average = 0.0;
        foreach (var mean in means) average += mean;
        average /= means.Count;

        var sumOfSquares = 0.0;
        foreach (var mean in means) sumOfSquares += (mean - average) * (mean - average);
        return (float)Math.Sqrt(sumOfSquares / (means.Count - 1));
    }

    private static float MeanIgnoringNaN(List<BenchmarkRunResult> runs, Func<BenchmarkRunResult, float> value)
    {
        var sum = 0.0;
        var count = 0;
        foreach (var run in runs)
        {
            var v = value(run);
            if (float.IsNaN(v)) continue;
            sum += v;
            count++;
        }
        return count > 0 ? (float)(sum / count) : float.NaN;
    }

    private static void WriteCsv(string path, IReadOnlyList<BenchmarkRunResult> results)
    {
        var sb = new StringBuilder();
        AppendRow(sb, CsvHeader);

        var row = new string[CsvHeader.Length];
        foreach (var result in results)
        {
            var pathFollowing = result.pathFollowing;
            var motionQuality = result.motionQuality;
            var cost = result.cost;

            var column = 0;
            row[column++] = result.method;
            row[column++] = result.path;
            // Blank, not 0, for a run that loaded each config's own seed.
            row[column++] = result.seeded ? result.seed.ToString(CultureInfo.InvariantCulture) : "";
            row[column++] = FormatFloat(result.completedLaps);
            row[column++] = result.timedOut ? "true" : "false";
            row[column++] = result.stoppedShort ? "true" : "false";
            row[column++] = FormatFloat(result.remainingDistance);
            row[column++] = result.framesTotal.ToString(CultureInfo.InvariantCulture);
            row[column++] = FormatFloat(result.durationSeconds);
            row[column++] = pathFollowing.framesEvaluated.ToString(CultureInfo.InvariantCulture);
            row[column++] = FormatFloat(pathFollowing.meanTrajectoryError);
            row[column++] = FormatFloat(pathFollowing.meanHeadingErrorDeg);
            row[column++] = FormatFloat(pathFollowing.maxHeadingErrorDeg);
            row[column++] = FormatFloat(pathFollowing.meanActualSpeed);
            row[column++] = FormatFloat(pathFollowing.targetSpeed);
            row[column++] = FormatFloat(pathFollowing.meanVelocityError);
            row[column++] = FormatFloat(motionQuality.footskatePerMeter);
            row[column++] = FormatFloat(motionQuality.meanFootskateSpeed);
            row[column++] = FormatFloat(motionQuality.contactFraction);
            row[column++] = FormatFloat(motionQuality.flagFootskatePerMeter);
            row[column++] = FormatFloat(motionQuality.flagMeanFootskateSpeed);
            row[column++] = FormatFloat(motionQuality.flagOnsetHeightError);
            row[column++] = FormatFloat(motionQuality.flagContactFraction);
            row[column++] = FormatFloat(motionQuality.contactFlagAgreement);
            row[column++] = FormatFloat(motionQuality.rootJerkMean);
            row[column++] = FormatFloat(motionQuality.rootJerkP95);
            row[column++] = FormatFloat(motionQuality.rootVerticalJerkMean);
            row[column++] = FormatFloat(motionQuality.rootVerticalJerkP95);
            row[column++] = FormatFloat(motionQuality.discontinuitiesPerSecond);
            row[column++] = FormatFloat(cost.applyMsMean);
            row[column++] = FormatFloat(cost.applyMsP50);
            row[column++] = FormatFloat(cost.applyMsP95);
            row[column++] = FormatFloat(cost.applyMsMax);
            row[column++] = FormatFloat(cost.gcBytesPerTick);
            row[column++] = result.recordingFile;
            row[column] = result.error;

            AppendRow(sb, row);
        }

        File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
    }

    private static string FormatFloat(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    private static void AppendRow(StringBuilder sb, string[] fields)
    {
        for (var i = 0; i < fields.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(CsvEscape(fields[i]));
        }
        sb.Append('\n');
    }

    private static string CsvEscape(string field)
    {
        if (string.IsNullOrEmpty(field)) return string.Empty;
        if (field.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0) return field;
        return "\"" + field.Replace("\"", "\"\"") + "\"";
    }

    private static void WriteJson(
        string path,
        SynthesisBenchmarkConfig config,
        string configAssetPath,
        IReadOnlyList<BenchmarkRunResult> results)
    {
        var resultsArray = new BenchmarkRunResult[results.Count];
        for (var i = 0; i < results.Count; i++) resultsArray[i] = results[i];

        var report = new BenchmarkReport
        {
            runTimestampUtc = DateTime.UtcNow.ToString("o"),
            unityVersion = Application.unityVersion,
            deviceName = SystemInfo.deviceName,
            processorType = SystemInfo.processorType,
            processorCount = SystemInfo.processorCount,
            configAssetPath = configAssetPath,
            lapsRequired = config.lapsRequired,
            settleTime = config.settleTime,
            maxRunSeconds = config.maxRunSeconds,
            restTimeout = config.restTimeout,
            synthesisFrameRate = config.synthesisFrameRate,
            fixedTimestep = config.fixedTimestep,
            speedSmoothingWindow = config.speedSmoothingWindow,
            results = resultsArray
        };

        File.WriteAllText(path, JsonUtility.ToJson(report, true), new UTF8Encoding(false));
    }

    private static void LogSummary(IReadOnlyList<BenchmarkRunResult> results)
    {
        var sb = new StringBuilder();
        sb.Append("Benchmark results:\n");
        sb.Append(string.Format(
            CultureInfo.InvariantCulture,
            "{0,-20} {1,-20} {2,8} {3,10} {4,10} {5,10} {6,10} {7,10}\n",
            "method", "path", "laps", "trajErr", "headErr", "footskate", "jerkP95", "applyMs"));

        foreach (var result in results)
        {
            var laps = result.completedLaps.ToString("F2", CultureInfo.InvariantCulture);
            if (result.timedOut) laps += "*";
            else if (result.stoppedShort) laps += "~";

            sb.Append(string.Format(
                CultureInfo.InvariantCulture,
                "{0,-20} {1,-20} {2,8} {3,10:F4} {4,10:F3} {5,10:F4} {6,10:F1} {7,10:F3}\n",
                result.seeded ? $"{result.method}_s{result.seed}" : result.method,
                result.path,
                laps,
                result.pathFollowing.meanTrajectoryError,
                result.pathFollowing.meanHeadingErrorDeg,
                result.motionQuality.footskatePerMeter,
                result.motionQuality.rootJerkP95,
                result.cost.applyMsMean));
        }

        sb.Append("(* = timed out before completing required laps, ~ = came to rest short of the end of an open path)");
        Debug.Log(sb.ToString());
    }

    /// <summary>
    /// Top-level shape of results.json: JsonUtility cannot serialise a bare array. The machine
    /// context is there because wall-clock cost only compares within one machine.
    /// </summary>
    [Serializable]
    private class BenchmarkReport
    {
        public string runTimestampUtc;
        public string unityVersion;
        public string deviceName;
        public string processorType;
        public int processorCount;
        public string configAssetPath;

        public float lapsRequired;
        public float settleTime;
        public float maxRunSeconds;
        public float restTimeout;
        public float synthesisFrameRate;
        public bool fixedTimestep;
        public float speedSmoothingWindow;

        public BenchmarkRunResult[] results;
    }
}
}
