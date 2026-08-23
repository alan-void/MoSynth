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
        "method", "path", "completedLaps", "timedOut", "framesTotal", "durationSeconds",
        "framesEvaluated", "meanTrajectoryError", "meanHeadingErrorDeg", "maxHeadingErrorDeg",
        "meanActualSpeed", "targetSpeed", "meanVelocityError", "footskatePerMeter",
        "meanFootskateSpeed", "contactFraction", "rootJerkMean", "rootJerkP95",
        "discontinuitiesPerSecond", "applyMsMean", "applyMsP50", "applyMsP95", "applyMsMax",
        "gcBytesPerTick", "recordingFile", "error"
    };

    /// <summary>
    /// Writes results.csv and results.json into <paramref name="outputDirectory"/> and logs a
    /// summary table. Returns the absolute path of results.csv.
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
        LogSummary(results);

        return csvPath;
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

            row[0] = result.method;
            row[1] = result.path;
            row[2] = FormatFloat(result.completedLaps);
            row[3] = result.timedOut ? "true" : "false";
            row[4] = result.framesTotal.ToString(CultureInfo.InvariantCulture);
            row[5] = FormatFloat(result.durationSeconds);
            row[6] = pathFollowing.framesEvaluated.ToString(CultureInfo.InvariantCulture);
            row[7] = FormatFloat(pathFollowing.meanTrajectoryError);
            row[8] = FormatFloat(pathFollowing.meanHeadingErrorDeg);
            row[9] = FormatFloat(pathFollowing.maxHeadingErrorDeg);
            row[10] = FormatFloat(pathFollowing.meanActualSpeed);
            row[11] = FormatFloat(pathFollowing.targetSpeed);
            row[12] = FormatFloat(pathFollowing.meanVelocityError);
            row[13] = FormatFloat(motionQuality.footskatePerMeter);
            row[14] = FormatFloat(motionQuality.meanFootskateSpeed);
            row[15] = FormatFloat(motionQuality.contactFraction);
            row[16] = FormatFloat(motionQuality.rootJerkMean);
            row[17] = FormatFloat(motionQuality.rootJerkP95);
            row[18] = FormatFloat(motionQuality.discontinuitiesPerSecond);
            row[19] = FormatFloat(cost.applyMsMean);
            row[20] = FormatFloat(cost.applyMsP50);
            row[21] = FormatFloat(cost.applyMsP95);
            row[22] = FormatFloat(cost.applyMsMax);
            row[23] = FormatFloat(cost.gcBytesPerTick);
            row[24] = result.recordingFile;
            row[25] = result.error;

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
            var laps = result.timedOut
                ? result.completedLaps.ToString("F2", CultureInfo.InvariantCulture) + "*"
                : result.completedLaps.ToString("F2", CultureInfo.InvariantCulture);

            sb.Append(string.Format(
                CultureInfo.InvariantCulture,
                "{0,-20} {1,-20} {2,8} {3,10:F4} {4,10:F3} {5,10:F4} {6,10:F1} {7,10:F3}\n",
                result.method,
                result.path,
                laps,
                result.pathFollowing.meanTrajectoryError,
                result.pathFollowing.meanHeadingErrorDeg,
                result.motionQuality.footskatePerMeter,
                result.motionQuality.rootJerkP95,
                result.cost.applyMsMean));
        }

        sb.Append("(* = timed out before completing required laps)");
        Debug.Log(sb.ToString());
    }

    /// <summary>
    /// Top-level shape of results.json. JsonUtility cannot serialise a bare array, so the run's
    /// results live inside this wrapper alongside the machine/version context needed to interpret
    /// wall-clock cost numbers that only mean anything relative to one another on one machine.
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
        public float synthesisFrameRate;
        public bool fixedTimestep;
        public float speedSmoothingWindow;

        public BenchmarkRunResult[] results;
    }
}
}
