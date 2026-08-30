using System;
using System.IO;
using UnityEngine;

namespace MotionField
{
/// <summary>Where a machine keeps its CPython library and the virtual environment to import from.</summary>
[Serializable]
public struct PythonPaths
{
    /// <summary>Full path to the CPython shared library, e.g. <c>.../python313.dll</c>, or empty.</summary>
    public string pythonDllPath;

    /// <summary>Virtual environment whose site-packages holds numpy, scipy and torch, or empty.</summary>
    public string pythonVenvPath;

    /// <summary>The same paths with nulls replaced by empty strings, which is what callers compare.</summary>
    public PythonPaths Normalized() => new()
    {
        pythonDllPath = pythonDllPath ?? "",
        pythonVenvPath = pythonVenvPath ?? ""
    };
}

/// <summary>
/// Project-wide store for <see cref="PythonPaths"/>, held per user rather than in a project asset.
/// Edited through <c>Project Settings &gt; MoSynth &gt; Python</c>.
/// </summary>
/// <remarks>
/// An interpreter's location is a property of a machine, so a path serialized into a shared asset
/// names someone else's drive on every other machine. <c>UserSettings/</c> is gitignored by the
/// standard Unity .gitignore, so this file never travels; it is plain JSON so shell and Python
/// tooling can read the same venv the Editor uses. The environment variables in
/// <see cref="PythonRuntime"/> still win, since they also reach a build machine that has no project
/// folder to read.
/// </remarks>
public static class PythonPathSettings
{
    /// <summary>Location of the settings file, relative to the project folder.</summary>
    public const string ProjectRelativePath = "UserSettings/MoSynthPython.json";

    /// <summary>Both paths unset, which is what every failed read comes back as.</summary>
    public static readonly PythonPaths Unset = new PythonPaths().Normalized();

    private static PythonPaths _cached;
    private static bool _loaded;

    /// <summary>Absolute path of the settings file, which need not exist.</summary>
    public static string FilePath =>
        Path.GetFullPath(Path.Combine(Application.dataPath, "..", ProjectRelativePath));

    /// <summary>
    /// The stored paths, read from disk once per domain. A missing or unreadable file reads as two
    /// empty paths, which is the same thing as far as every caller is concerned.
    /// </summary>
    public static PythonPaths Current
    {
        get
        {
            if (_loaded) return _cached;

            _cached = ReadFile();
            _loaded = true;
            return _cached;
        }
    }

    /// <summary>Writes <paramref name="paths"/> to <see cref="FilePath"/> and updates the cache.</summary>
    public static void Save(PythonPaths paths)
    {
        paths = paths.Normalized();
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath) ?? "");
        File.WriteAllText(FilePath, ToJson(paths));
        _cached = paths;
        _loaded = true;
    }

    /// <summary>Drops the cache, so the next read picks up a file edited outside the Editor.</summary>
    public static void Reload() => _loaded = false;

    /// <summary>The JSON form written to <see cref="FilePath"/>.</summary>
    public static string ToJson(PythonPaths paths) => JsonUtility.ToJson(paths.Normalized(), true);

    /// <summary>
    /// Reads the JSON form. Anything unparseable reads as two empty paths rather than throwing:
    /// this file is hand-editable, and a typo in it should leave a message about where Python is,
    /// not an exception out of an unrelated caller.
    /// </summary>
    public static PythonPaths Parse(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return Unset;

        try
        {
            return JsonUtility.FromJson<PythonPaths>(json).Normalized();
        }
        catch (ArgumentException)
        {
            return Unset;
        }
    }

    private static PythonPaths ReadFile()
    {
        var path = FilePath;
        if (!File.Exists(path)) return Unset;

        try
        {
            return Parse(File.ReadAllText(path));
        }
        catch (IOException e)
        {
            Debug.LogWarning($"Could not read \"{path}\" ({e.Message}); treating the Python paths as unset.");
            return Unset;
        }
    }
}
}
