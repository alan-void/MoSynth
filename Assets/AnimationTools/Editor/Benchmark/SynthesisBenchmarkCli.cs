using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Command-line entry point for a benchmark sweep:
/// <c>-executeMethod AnimationTools.Editor.SynthesisBenchmarkCli.Run</c>.
/// </summary>
/// <remarks>
/// The invocation must NOT pass <c>-quit</c>. A sweep runs in play mode, so this method returns
/// long before it finishes; <c>-quit</c> would tear the Editor down mid-run. Batchmode Unity stays
/// alive after <c>-executeMethod</c> returns on its own, and
/// <see cref="SynthesisBenchmarkDriver"/> calls <see cref="EditorApplication.Exit"/> with the
/// sweep's status code once the report is written. See <c>Tools/run-benchmark.ps1</c>.
/// </remarks>
public static class SynthesisBenchmarkCli
{
    private const string ConfigArgument = "-benchmarkConfig";
    private const string OutputArgument = "-benchmarkOutput";

    public static void Run()
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

            var output = GetArgument(OutputArgument);
            if (string.IsNullOrEmpty(output))
            {
                output = Path.Combine(config.outputDirectory, SynthesisBenchmarkLauncher.TimestampFolderName());
            }

            var outputDirectory = SynthesisBenchmarkLauncher.ResolveOutputDirectory(output);

            // Only headless runs should take the Editor down with them; a developer running this
            // from a live Editor session wants their Editor back, not a process exit.
            var headless = Application.isBatchMode;

            if (!SynthesisBenchmarkLauncher.Launch(config, outputDirectory, headless))
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
    /// it does not recognise, which is what lets a sweep be parameterised this way at all.
    /// </summary>
    private static string GetArgument(string name)
    {
        var args = Environment.GetCommandLineArgs();
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        }

        return null;
    }

    private static void Fail(string message)
    {
        Debug.LogError($"[Benchmark] {message}");

        // Nothing downstream will run to set an exit code, so a headless invocation has to stop
        // here or it would sit idle until the harness times out.
        if (Application.isBatchMode) EditorApplication.Exit(1);
    }
}
}
