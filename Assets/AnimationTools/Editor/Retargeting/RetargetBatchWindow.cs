using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace AnimationTools.Editor
{
/// <summary>
/// Launches the headless Blender batch that retargets a folder of BVH files onto the shared
/// target rig and writes one FBX. The retargeting itself lives in
/// Tools/Retargeting/batch_retarget.py; this only starts it and refreshes the AssetDatabase.
/// </summary>
public class RetargetBatchWindow : EditorWindow
{
    private const string BlenderPathKey = "MoSynth.Retargeting.BlenderPath";
    private const string ScriptPath = "Tools/Retargeting/batch_retarget.py";

    // Blender exits 0 even when the script raises, and even when it fails to parse at all, so a
    // clean exit code is not evidence that the batch ran. The script's closing line is.
    private const string CompletionLine = "[retarget] done";

    private static readonly string[] BlenderProbePaths =
    {
        @"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        @"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    };

    // A run outlives any one window, so the process and its output queue are static; reopening
    // or closing the window must not orphan a batch that is still going.
    private static Process _process;
    private static readonly ConcurrentQueue<string> Output = new();
    private static string _outputPath;
    private static bool _sawCompletion;

    private RetargetBatchSettings _settings;
    private string _blenderPath;

    [MenuItem("MoSynth/Retargeting/Run Batch...")]
    private static void Open() => GetWindow<RetargetBatchWindow>(true, "Retarget Batch");

    private void OnEnable()
    {
        _blenderPath = EditorPrefs.GetString(BlenderPathKey, string.Empty);
        if (string.IsNullOrEmpty(_blenderPath)) _blenderPath = ProbeForBlender();
    }

    private static string ProbeForBlender()
    {
        foreach (var path in BlenderProbePaths)
        {
            if (File.Exists(path)) return path;
        }

        return string.Empty;
    }

    private void OnGUI()
    {
        _settings = (RetargetBatchSettings)EditorGUILayout.ObjectField(
            "Settings", _settings, typeof(RetargetBatchSettings), false);

        EditorGUI.BeginChangeCheck();
        using (new EditorGUILayout.HorizontalScope())
        {
            _blenderPath = EditorGUILayout.TextField("Blender", _blenderPath);
            if (GUILayout.Button("...", GUILayout.Width(30)))
            {
                var picked = EditorUtility.OpenFilePanel("Blender executable", "", "exe");
                if (!string.IsNullOrEmpty(picked)) _blenderPath = picked;
            }
        }

        if (EditorGUI.EndChangeCheck()) EditorPrefs.SetString(BlenderPathKey, _blenderPath);

        EditorGUILayout.HelpBox(
            "Retargeting is configured in the setup .blend, not here. See " +
            "Tools/Retargeting/README.md.", MessageType.Info);

        if (_process != null)
        {
            EditorGUILayout.LabelField("Running...", EditorStyles.boldLabel);
            if (GUILayout.Button("Cancel")) Cancel();
            return;
        }

        using (new EditorGUI.DisabledScope(_settings == null))
        {
            if (GUILayout.Button("Run Batch")) Run(_settings, _blenderPath);
        }
    }

    private static void Run(RetargetBatchSettings settings, string blenderPath)
    {
        var projectRoot = Path.GetDirectoryName(Application.dataPath);

        if (!File.Exists(blenderPath))
        {
            EditorUtility.DisplayDialog("Retarget Batch",
                string.IsNullOrEmpty(blenderPath)
                    ? "No Blender executable set. Pick blender.exe with the \"...\" button."
                    : $"No Blender executable at \"{blenderPath}\".", "OK");
            return;
        }

        var script = Path.Combine(projectRoot, ScriptPath);
        if (!File.Exists(script))
        {
            EditorUtility.DisplayDialog("Retarget Batch", $"The batch script is missing: {script}", "OK");
            return;
        }

        var blend = Path.Combine(projectRoot, settings.setupBlendPath);
        if (!File.Exists(blend))
        {
            EditorUtility.DisplayDialog("Retarget Batch",
                $"No setup blend at \"{settings.setupBlendPath}\".", "OK");
            return;
        }

        var bvhFolder = Path.Combine(projectRoot, settings.bvhFolderPath);
        if (!Directory.Exists(bvhFolder))
        {
            EditorUtility.DisplayDialog("Retarget Batch",
                $"No BVH folder at \"{settings.bvhFolderPath}\".", "OK");
            return;
        }

        _outputPath = Path.Combine(projectRoot, settings.outputFbxPath);

        // --factory-startup is deliberately absent: the batch's second stage is the Rokoko
        // addon, which a factory startup would not load.
        var arguments =
            $"--background {Quote(blend)} --python {Quote(script)} -- " +
            $"--bvh-dir {Quote(bvhFolder)} --pattern {Quote(settings.filePattern)} " +
            $"--out {Quote(_outputPath)} --simplify {settings.simplifyFactor.ToString("R", CultureInfo.InvariantCulture)}" +
            (settings.clipLimit > 0 ? $" --limit {settings.clipLimit}" : string.Empty);

        _process = new Process
        {
            StartInfo = new ProcessStartInfo(blenderPath, arguments)
            {
                WorkingDirectory = projectRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            },
            EnableRaisingEvents = true,
        };

        // Output arrives on a background thread, and Debug.Log is main-thread only.
        _process.OutputDataReceived += (_, e) => Enqueue(e.Data);
        _process.ErrorDataReceived += (_, e) => Enqueue(e.Data);

        _sawCompletion = false;
        _process.Start();
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();

        EditorApplication.update += Poll;
        Debug.Log($"Retarget batch started: {settings.filePattern} from {settings.bvhFolderPath}");
    }

    private static void Enqueue(string line)
    {
        if (string.IsNullOrEmpty(line) || !line.StartsWith("[retarget]")) return;

        if (line == CompletionLine) _sawCompletion = true;
        Output.Enqueue(line);
    }

    private static string Quote(string value) => "\"" + value + "\"";

    private static void Poll()
    {
        while (Output.TryDequeue(out var line)) Debug.Log(line);

        if (_process == null || !_process.HasExited) return;

        var exitCode = _process.ExitCode;
        _process.Dispose();
        _process = null;
        EditorApplication.update -= Poll;

        if (exitCode == 0 && _sawCompletion)
        {
            AssetDatabase.Refresh();
            Debug.Log($"Retarget batch finished: {_outputPath}");
        }
        else
        {
            Debug.LogError(
                $"Retarget batch failed (exit code {exitCode}). The lines above say what went wrong; " +
                "a setup problem names what the blend was expected to contain.");
        }

        foreach (var window in Resources.FindObjectsOfTypeAll<RetargetBatchWindow>()) window.Repaint();
    }

    private static void Cancel()
    {
        if (_process == null) return;

        try
        {
            _process.Kill();
        }
        catch (InvalidOperationException)
        {
            // Exited between the check and the kill; Poll cleans up either way.
        }
    }
}
}
