using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using Debug = UnityEngine.Debug;

namespace AnimationTools.Editor
{
/// <summary>
/// Runs <c>python -m autotag.annotate</c> on a run folder as a child process and follows its
/// progress lines. Poll <see cref="Update"/> from the main thread.
/// </summary>
/// <remarks>
/// A child process rather than PythonNET: an annotation run makes network calls for minutes, and
/// holding the GIL on the main thread for that long would freeze the Editor.
/// </remarks>
public sealed class AutoTagAnnotateProcess : IDisposable
{
    public const string ApiKeyVariable = "GEMINI_API_KEY";
    public const string Module = "autotag.annotate";

    /// <summary>Exit code: every clip succeeded.</summary>
    public const int ExitSuccess = 0;

    /// <summary>Exit code: some clips failed, and results.json was still written.</summary>
    public const int ExitPartial = 2;

    private readonly Process _process;
    private readonly ConcurrentQueue<string> _stdout = new();
    private readonly StringBuilder _stderr = new();

    public string RunFolder { get; }
    public int Index { get; private set; }
    public int Total { get; private set; }
    public string CurrentClip { get; private set; } = "";
    public long InputTokens { get; private set; }
    public long OutputTokens { get; private set; }
    public int ClipErrors { get; private set; }
    public bool HasExited { get; private set; }
    public int ExitCode { get; private set; } = -1;

    /// <summary>What the process wrote to stderr; complete once <see cref="HasExited"/>.</summary>
    public string Stderr
    {
        get
        {
            lock (_stderr) return _stderr.ToString();
        }
    }

    public float Progress => Total <= 0 ? 0f : Math.Min(1f, (float)Index / Total);

    private AutoTagAnnotateProcess(Process process, string runFolder)
    {
        _process = process;
        RunFolder = runFolder;
    }

    /// <summary>
    /// The API key from this process's environment, else the user's, which also finds one set with
    /// setx after Unity started. Null when neither has one.
    /// </summary>
    public static string FindApiKey()
    {
        var key = Environment.GetEnvironmentVariable(ApiKeyVariable);
        if (!string.IsNullOrWhiteSpace(key)) return key;

        try
        {
            key = Environment.GetEnvironmentVariable(ApiKeyVariable, EnvironmentVariableTarget.User);
            return string.IsNullOrWhiteSpace(key) ? null : key;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>The venv's interpreter, from the same sources <see cref="PythonRuntime"/> uses; empty if none.</summary>
    public static string FindPythonExecutable(out string error)
    {
        var venv = PythonRuntime.ResolveVenv();
        if (string.IsNullOrWhiteSpace(venv))
        {
            error = "No Python virtual environment is set. Set it in Project Settings > MoSynth > Python, " +
                    $"or in the {PythonRuntime.PythonVenvVariable} environment variable.";
            return "";
        }

        foreach (var candidate in new[]
                 {
                     Path.Combine(venv, "Scripts", "python.exe"),
                     Path.Combine(venv, "bin", "python3"),
                     Path.Combine(venv, "bin", "python")
                 })
        {
            if (!File.Exists(candidate)) continue;

            error = null;
            return candidate;
        }

        error = $"No Python interpreter under \"{venv}\" (looked in Scripts/ and bin/).";
        return "";
    }

    /// <summary>Launches the annotator on <paramref name="runFolder"/>; null with an error if it cannot start.</summary>
    public static AutoTagAnnotateProcess Start(string runFolder, out string error)
    {
        var python = FindPythonExecutable(out error);
        if (string.IsNullOrEmpty(python)) return null;

        var apiKey = FindApiKey();
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            error = $"{ApiKeyVariable} is not set.";
            return null;
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = python,
            Arguments = $"-m {Module} \"{Path.GetFullPath(runFolder)}\"",
            WorkingDirectory = PythonRuntime.ScriptsFolder,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
        startInfo.EnvironmentVariables[ApiKeyVariable] = apiKey;
        startInfo.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
        startInfo.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

        var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        var runner = new AutoTagAnnotateProcess(process, runFolder);
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data != null) runner._stdout.Enqueue(e.Data);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data == null) return;
            lock (runner._stderr) runner._stderr.AppendLine(e.Data);
        };

        try
        {
            process.Start();
        }
        catch (Exception e)
        {
            error = $"Could not start \"{python}\": {e.Message}";
            process.Dispose();
            return null;
        }

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        error = null;
        return runner;
    }

    /// <summary>Consumes output received so far and notices exit. True once the process has exited.</summary>
    public bool Update()
    {
        if (HasExited) return true;

        DrainStdout();

        if (!_process.HasExited) return false;

        // The parameterless wait also waits for the redirected streams to reach end of file.
        _process.WaitForExit();
        DrainStdout();

        ExitCode = _process.ExitCode;
        HasExited = true;
        return true;
    }

    public void Kill()
    {
        if (HasExited) return;

        try
        {
            if (!_process.HasExited) _process.Kill();
        }
        catch (InvalidOperationException)
        {
            // Exited between the check and the kill.
        }
    }

    private void DrainStdout()
    {
        while (_stdout.TryDequeue(out var line))
        {
            if (!TryHandleProtocolLine(line)) Debug.Log($"[AutoTag] {line}");
        }
    }

    private bool TryHandleProtocolLine(string line)
    {
        var parsed = ProtocolLine.Parse(line);
        switch (parsed.Kind)
        {
            case ProtocolLine.LineKind.Progress:
                Index = parsed.Index;
                Total = parsed.Total;
                CurrentClip = parsed.Text;
                return true;

            case ProtocolLine.LineKind.Usage:
                InputTokens += parsed.InputTokens;
                OutputTokens += parsed.OutputTokens;
                return true;

            case ProtocolLine.LineKind.ClipError:
                ClipErrors++;
                Debug.LogWarning($"[AutoTag] Clip {parsed.Guid} failed: {parsed.Text}");
                return true;

            case ProtocolLine.LineKind.Done:
                Index = Total;
                Debug.Log($"[AutoTag] Annotation done: {parsed.Index} succeeded, {parsed.Total} failed.");
                return true;

            default:
                return false;
        }
    }

    public void Dispose()
    {
        Kill();
        _process.Dispose();
    }

    /// <summary>One parsed stdout line of the annotator's progress protocol.</summary>
    public readonly struct ProtocolLine
    {
        public enum LineKind
        {
            Other,
            Progress,   // PROGRESS <index> <total> <clipName>
            Usage,      // USAGE <guid> <inputTokens> <outputTokens>
            ClipError,  // CLIPERROR <guid> <message>
            Done        // DONE <succeeded> <failed>
        }

        public readonly LineKind Kind;

        /// <summary>PROGRESS index, or DONE succeeded count.</summary>
        public readonly int Index;

        /// <summary>PROGRESS total, or DONE failed count.</summary>
        public readonly int Total;

        public readonly string Guid;
        public readonly long InputTokens;
        public readonly long OutputTokens;

        /// <summary>PROGRESS clip name, or CLIPERROR message.</summary>
        public readonly string Text;

        private ProtocolLine(LineKind kind, int index = 0, int total = 0, string guid = null,
            long inputTokens = 0, long outputTokens = 0, string text = null)
        {
            Kind = kind;
            Index = index;
            Total = total;
            Guid = guid;
            InputTokens = inputTokens;
            OutputTokens = outputTokens;
            Text = text;
        }

        /// <summary>Parses one line; anything malformed is <see cref="LineKind.Other"/> and just logged.</summary>
        public static ProtocolLine Parse(string line)
        {
            if (string.IsNullOrEmpty(line)) return default;

            var parts = line.Split(new[] { ' ' }, 4);
            switch (parts[0])
            {
                case "PROGRESS" when parts.Length >= 3 && TryInt(parts[1], out var index) &&
                                     TryInt(parts[2], out var total):
                    return new ProtocolLine(LineKind.Progress, index, total, text: parts.Length > 3 ? parts[3] : "");

                case "USAGE" when parts.Length >= 4 && TryLong(parts[2], out var input) &&
                                  TryLong(parts[3], out var output):
                    return new ProtocolLine(LineKind.Usage, guid: parts[1], inputTokens: input, outputTokens: output);

                case "CLIPERROR" when parts.Length >= 2:
                    var message = line.Length > parts[0].Length + parts[1].Length + 2
                        ? line.Substring(parts[0].Length + parts[1].Length + 2)
                        : "";
                    return new ProtocolLine(LineKind.ClipError, guid: parts[1], text: message);

                case "DONE" when parts.Length >= 3 && TryInt(parts[1], out var succeeded) &&
                                 TryInt(parts[2], out var failed):
                    return new ProtocolLine(LineKind.Done, succeeded, failed);

                default:
                    return default;
            }
        }

        private static bool TryInt(string text, out int value) =>
            int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);

        private static bool TryLong(string text, out long value) =>
            long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }
}
}
