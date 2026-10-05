using System;
using System.IO;
using Python.Runtime;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Shared CPython bootstrap for every synthesis method that runs Python.
/// </summary>
/// <remarks>
/// The interpreter is process-wide and configurable only once, so every caller comes through here
/// rather than calling <see cref="PythonEngine.Initialize()"/> with its own paths.
/// </remarks>
public static class PythonRuntime
{
    /// <summary>
    /// Environment variable naming the CPython shared library, checked before
    /// <see cref="PythonPathSettings"/>.
    /// </summary>
    /// <remarks>
    /// Wins over the settings file because it also reaches a machine with no project folder to read,
    /// such as a build agent or a player.
    /// </remarks>
    public const string PythonDllVariable = "MOSYNTH_PYTHON_DLL";

    /// <summary>Environment variable naming the virtual environment. See <see cref="PythonDllVariable"/>.</summary>
    public const string PythonVenvVariable = "MOSYNTH_PYTHON_VENV";

    /// <summary>Where a resolved path came from, which is what the settings UI reports.</summary>
    public enum PathSource
    {
        /// <summary>Neither source names one.</summary>
        Unset,

        /// <summary>The <see cref="PythonDllVariable"/> / <see cref="PythonVenvVariable"/> variable.</summary>
        Environment,

        /// <summary><see cref="PythonPathSettings"/>.</summary>
        Settings
    }

    private static bool _pythonDllAssigned;
    private static string _scriptsFolder;

    /// <summary>The repository's Python folder, which is where our modules are imported from.</summary>
    public static string ScriptsFolder =>
        _scriptsFolder ??= Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Python"));

    /// <summary>
    /// The CPython shared library that will actually be used. Empty means neither source names one,
    /// and pythonnet falls back to PYTHONNET_PYDLL on its own.
    /// </summary>
    public static string ResolvePythonDll(out PathSource source) =>
        Resolve(PythonDllVariable, PythonPathSettings.Current.pythonDllPath, out source);

    /// <inheritdoc cref="ResolvePythonDll(out PathSource)"/>
    public static string ResolvePythonDll() => ResolvePythonDll(out _);

    /// <summary>
    /// The virtual environment that will actually be used. Empty means neither source names one,
    /// and only the interpreter's own site-packages is importable.
    /// </summary>
    public static string ResolveVenv(out PathSource source) =>
        Resolve(PythonVenvVariable, PythonPathSettings.Current.pythonVenvPath, out source);

    /// <inheritdoc cref="ResolveVenv(out PathSource)"/>
    public static string ResolveVenv() => ResolveVenv(out _);

    /// <summary>
    /// Picks between the environment variable and the settings file, treating a blank value in
    /// either as absent.
    /// </summary>
    public static string Resolve(string variable, string settingsPath, out PathSource source)
    {
        var fromEnvironment = Environment.GetEnvironmentVariable(variable);
        if (!string.IsNullOrWhiteSpace(fromEnvironment))
        {
            source = PathSource.Environment;
            return fromEnvironment;
        }

        source = string.IsNullOrWhiteSpace(settingsPath) ? PathSource.Unset : PathSource.Settings;
        return settingsPath ?? "";
    }

    /// <summary>
    /// Start CPython if it is not already running and make the project's Python modules importable,
    /// using the paths this machine is configured with. Safe to call repeatedly.
    /// </summary>
    /// <remarks>
    /// Only the first call of the process can choose the interpreter -- pythonnet throws if the DLL
    /// is changed after the engine starts -- so changing the paths needs a domain reload.
    /// </remarks>
    public static void EnsureInitialized()
    {
        var pythonDllPath = ResolvePythonDll();
        var venvPath = ResolveVenv();

        if (!PythonEngine.IsInitialized)
        {
            if (!_pythonDllAssigned && !string.IsNullOrWhiteSpace(pythonDllPath))
            {
                if (!File.Exists(pythonDllPath))
                {
                    throw new FileNotFoundException(
                        $"Python DLL not found at '{pythonDllPath}'. Set it in " +
                        $"Project Settings > MoSynth > Python, or in the {PythonDllVariable} " +
                        "environment variable.",
                        pythonDllPath);
                }

                Runtime.PythonDLL = pythonDllPath;
                _pythonDllAssigned = true;
            }

            PythonEngine.Initialize();
        }

        using (Py.GIL())
        {
            if (!string.IsNullOrWhiteSpace(venvPath))
            {
                var sitePackages = Path.Combine(venvPath, "Lib", "site-packages");
                if (!Directory.Exists(sitePackages))
                {
                    throw new DirectoryNotFoundException(
                        $"No site-packages under '{venvPath}'. Set it in " +
                        $"Project Settings > MoSynth > Python, or in the {PythonVenvVariable} " +
                        "environment variable.");
                }

                dynamic site = Py.Import("site");
                site.addsitedir(sitePackages);
            }

            dynamic sys = Py.Import("sys");
            AppendToPath(sys, ScriptsFolder);
        }
    }

    private static void AppendToPath(dynamic sys, string folder)
    {
        foreach (dynamic entry in sys.path)
        {
            if (string.Equals(entry.ToString(), folder, StringComparison.OrdinalIgnoreCase)) return;
        }

        sys.path.append(folder);
    }

    /// <summary>
    /// Import a module from the project's Python folder.
    /// </summary>
    /// <param name="moduleName">Dotted module path from the Python folder, e.g. <c>pfnn.runtime</c>.</param>
    /// <param name="reload">
    /// Pick up edits to the .py files without restarting Unity, by dropping the whole project
    /// module graph first -- see <see cref="InvalidateProjectModules"/>. Call it once before a
    /// group of related imports rather than on each one, so they all end up sharing one generation
    /// of the code.
    /// </param>
    public static dynamic Import(string moduleName, bool reload = false)
    {
        if (reload) InvalidateProjectModules();
        return Py.Import(moduleName);
    }

    /// <summary>
    /// A Python list of strings as a C# array, or null for Python <c>None</c>. Call under the GIL.
    /// </summary>
    /// <remarks>Member access through <c>dynamic</c> hands <c>None</c> back as a C# null, so test both.</remarks>
    public static string[] ToStringArrayOrNull(dynamic value)
    {
        if (value == null) return null;
        return ((PyObject)value).IsNone() ? null : (string[])value;
    }

    /// <summary>
    /// Forget every already-imported module that lives under <see cref="ScriptsFolder"/>, so the
    /// next import re-reads them from disk. Returns how many were dropped.
    /// </summary>
    /// <remarks>
    /// Dropping the whole graph, rather than <c>importlib.reload</c> per module, keeps dependencies
    /// fresh and gives later imports one consistent generation of classes. Objects created earlier
    /// keep their old classes, so call this only where everything downstream is about to be rebuilt.
    /// </remarks>
    public static int InvalidateProjectModules()
    {
        using PyModule scope = Py.CreateScope();
        scope.Set("root", ScriptsFolder.ToPython());
        scope.Exec(InvalidateScript);
        return scope.Get<int>("dropped");
    }

    // Kept as source rather than a .py file in ScriptsFolder: a module that purges the project's
    // modules should not be one of the modules it purges.
    private const string InvalidateScript = @"
import os
import sys

prefix = os.path.normcase(os.path.abspath(root)) + os.sep
stale = []
for name, module in list(sys.modules.items()):
    path = getattr(module, '__file__', None)
    if not path:
        continue
    if os.path.normcase(os.path.abspath(path)).startswith(prefix):
        stale.append(name)

for name in stale:
    del sys.modules[name]

dropped = len(stale)
";

    /// <summary>
    /// A Python value as JSON text, for logs a later script reads back. Call under the GIL.
    /// </summary>
    /// <remarks>
    /// NaN and infinity become null, because strict JSON has no spelling for them; numpy and torch
    /// values go through <c>tolist()</c>, and anything else unencodable through <c>str()</c>.
    /// </remarks>
    public static string ToJson(PyObject value)
    {
        using PyModule scope = Py.CreateScope();
        scope.Set("value", value);
        scope.Exec(ToJsonScript);
        return scope.Get<string>("text");
    }

    private const string ToJsonScript = @"
import json
import math


def _clean(v):
    if hasattr(v, 'tolist'):
        v = v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


text = json.dumps(_clean(value), default=str)
";
}
}
