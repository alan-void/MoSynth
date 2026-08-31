using System;
using System.IO;
using Python.Runtime;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// <c>Project Settings &gt; MoSynth &gt; Python</c>: the CPython library and virtual environment
/// this machine should use, and which of the two sources is actually supplying each.
/// </summary>
public static class PythonSettingsProvider
{
    private const string SettingsPath = "Project/MoSynth/Python";

    [SettingsProvider]
    public static SettingsProvider Create() =>
        new(SettingsPath, SettingsScope.Project)
        {
            label = "Python",
            keywords = new[] { "python", "cpython", "venv", "virtual environment", "pythonnet", "dll" },
            guiHandler = _ => Draw()
        };

    /// <summary>Opens the page, which is where every "Python is not set up" message points.</summary>
    public static void Open() => SettingsService.OpenProjectSettings(SettingsPath);

    private static void Draw()
    {
        var stored = PythonPathSettings.Current;
        var edited = stored;

        EditorGUILayout.HelpBox(
            "Stored per user in " + PythonPathSettings.ProjectRelativePath + ", which is gitignored. " +
            "An interpreter's location belongs to a machine, so these paths are deliberately not " +
            "part of any asset.", MessageType.None);

        EditorGUILayout.Space();

        edited.pythonDllPath = PathField(
            new GUIContent("CPython library",
                "Full path to python313.dll. Python 3.13 is required for pythonnet compatibility."),
            stored.pythonDllPath, PythonRuntime.PythonDllVariable, isFolder: false);

        edited.pythonVenvPath = PathField(
            new GUIContent("Virtual environment",
                "Folder whose Lib/site-packages holds numpy, scipy and torch."),
            stored.pythonVenvPath, PythonRuntime.PythonVenvVariable, isFolder: true);

        if (edited.pythonDllPath != stored.pythonDllPath || edited.pythonVenvPath != stored.pythonVenvPath)
        {
            PythonPathSettings.Save(edited);
        }

        EditorGUILayout.Space();

        if (GUILayout.Button("Start Python and report what it found"))
        {
            Verify();
        }

        // The interpreter is chosen once per process, so an edit made after something already
        // imported Python does not take effect until the domain is reloaded.
        if (PythonEngine.IsInitialized)
        {
            EditorGUILayout.HelpBox(
                "Python is already running in this session. Changing these paths takes effect after " +
                "a domain reload -- recompile, or restart the Editor.", MessageType.Info);
        }

        using (new EditorGUI.DisabledScope(true))
        {
            EditorGUILayout.TextField("Settings file", PythonPathSettings.FilePath);
            EditorGUILayout.TextField("Project modules", PythonRuntime.ScriptsFolder);
        }
    }

    /// <summary>
    /// Starts the interpreter and reports the version and where imports resolve from, so a wrong
    /// path is found here rather than in the middle of a training run.
    /// </summary>
    [MenuItem("MoSynth/Python/Verify Setup")]
    public static void Verify()
    {
        try
        {
            PythonRuntime.EnsureInitialized();
            using (Py.GIL())
            {
                dynamic sys = Py.Import("sys");
                var numpy = TryImportVersion("numpy");
                var torch = TryImportVersion("torch");
                Debug.Log($"Python {sys.version.ToString()}\n" +
                          $"executable: {sys.executable.ToString()}\n" +
                          $"numpy: {numpy}, torch: {torch}\n" +
                          $"project modules: {PythonRuntime.ScriptsFolder}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Python did not start: {e.Message}");
        }
    }

    private static string TryImportVersion(string moduleName)
    {
        try
        {
            dynamic module = Py.Import(moduleName);
            return module.__version__.ToString();
        }
        catch (PythonException)
        {
            return "not importable";
        }
    }

    /// <summary>
    /// One path row: the stored value, a browse button, and a line saying what will actually be
    /// used. Returns the possibly edited value.
    /// </summary>
    private static string PathField(GUIContent label, string stored, string variable, bool isFolder)
    {
        var effective = PythonRuntime.Resolve(variable, stored, out var source);

        EditorGUILayout.BeginHorizontal();
        var edited = EditorGUILayout.TextField(label, stored ?? "");
        if (GUILayout.Button("Browse…", GUILayout.Width(70)))
        {
            var picked = isFolder
                ? EditorUtility.OpenFolderPanel(label.text, StartFolder(stored), "")
                : EditorUtility.OpenFilePanel(label.text, StartFolder(stored), "dll");
            if (!string.IsNullOrEmpty(picked)) edited = picked;
        }

        EditorGUILayout.EndHorizontal();

        ++EditorGUI.indentLevel;
        EditorGUILayout.LabelField(" ", Describe(effective, source, variable, isFolder), EditorStyles.miniLabel);
        --EditorGUI.indentLevel;

        return edited;
    }

    private static string Describe(string effective, PythonRuntime.PathSource source, string variable,
        bool isFolder)
    {
        if (source == PythonRuntime.PathSource.Unset) return "Not set.";

        var origin = source == PythonRuntime.PathSource.Environment
            ? $"{variable} overrides this field: "
            : "";
        var exists = isFolder ? Directory.Exists(effective) : File.Exists(effective);
        return origin + effective + (exists ? "" : "  (does not exist)");
    }

    private static string StartFolder(string current) =>
        string.IsNullOrWhiteSpace(current) ? "" : Path.GetDirectoryName(current) ?? "";
}
}
