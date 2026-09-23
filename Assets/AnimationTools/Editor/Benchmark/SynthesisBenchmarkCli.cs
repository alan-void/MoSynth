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
/// The invocation must NOT pass <c>-quit</c>: the sweep runs in play mode long after this returns,
/// and <see cref="SynthesisBenchmarkDriver"/> calls <see cref="EditorApplication.Exit"/> itself.
/// See <c>Tools/run-benchmark.ps1</c>.
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

            // Only headless runs take the Editor down with them.
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
    /// it does not recognise, which is what makes custom ones possible.
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

        // Nothing downstream will set an exit code, so a headless run would otherwise sit idle.
        if (Application.isBatchMode) EditorApplication.Exit(1);
    }
}
}
