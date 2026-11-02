using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace AnimationTools.Editor
{
/// <summary>
/// Command-line entry point for a benchmark sweep:
/// <c>-executeMethod AnimationTools.Editor.SynthesisBenchmarkCli.Run</c>.
/// </summary>
/// <remarks>
/// The invocation must NOT pass <c>-quit</c>: the sweep runs in play mode long after this returns,
/// and <see cref="SynthesisBenchmarkDriver"/> calls <see cref="EditorApplication.Exit"/> itself.
/// See <c>Tools/run-benchmark.ps1</c>.
/// </remarks>
public static class SynthesisBenchmarkCli
{
    private const string ConfigArgument = "-benchmarkConfig";
    private const string OutputArgument = "-benchmarkOutput";

    public static void Run() => Run(null);

    /// <summary>
    /// <see cref="Run()"/>, with <paramref name="sweepSeeds"/> replacing the seed list of every method
    /// that loads a per-seed checkpoint; null keeps each method's own.
    /// </summary>
    public static void Run(IReadOnlyList<int> sweepSeeds)
    {
        try
        {
            var configPath = GetArgument(ConfigArgument);
            if (string.IsNullOrEmpty(configPath))
            {
                Fail($"{ConfigArgument} <path/to/config.asset> is required.");
                return;
            }

            var config = AssetDatabase.LoadAssetAtPath<SynthesisBenchmarkConfig>(configPath);
            if (config == null)
            {
                Fail($"No SynthesisBenchmarkConfig at \"{configPath}\".");
                return;
            }

            // Rendering without a device crashes the Editor natively rather than throwing.
            if (config.recordVideo && SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null)
            {
                Fail($"\"{config.name}\" records video, which needs a graphics device: launch without " +
                     "-nographics (run-benchmark.ps1 -Graphics).");
                return;
            }

            var output = GetArgument(OutputArgument);
            if (string.IsNullOrEmpty(output))
            {
                output = Path.Combine(config.outputDirectory, SynthesisBenchmarkLauncher.TimestampFolderName());
            }

            var outputDirectory = SynthesisBenchmarkLauncher.ResolveOutputDirectory(output);

            // Only headless runs take the Editor down with them.
            var headless = Application.isBatchMode;

            if (!SynthesisBenchmarkLauncher.Launch(config, outputDirectory, headless, sweepSeeds))
            {
                Fail("The sweep could not be started (see the errors above).");
            }
        }
        catch (Exception e)
        {
            Fail(e.ToString());
        }
    }

    /// <summary>
    /// Value of a <c>-name value</c> pair on Unity's command line, or null. Unity ignores arguments
    /// it does not recognise, which is what makes custom ones possible.
    /// </summary>
    public static string GetArgument(string name)
    {
        var args = Environment.GetCommandLineArgs();
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        }

        return null;
    }

    /// <summary>Logs a failure and, headless, exits with 1: nothing else will set an exit code.</summary>
    public static void Fail(string message)
    {
        Debug.LogError($"[Benchmark] {message}");

        // Nothing downstream will set an exit code, so a headless run would otherwise sit idle.
        if (Application.isBatchMode) EditorApplication.Exit(1);
    }
}
}
