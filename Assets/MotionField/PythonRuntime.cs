using System;
using System.IO;
using Python.Runtime;
using UnityEngine;

namespace MotionField
{
/// <summary>
/// Shared CPython bootstrap for everything on the MotionField side.
///
/// The interpreter is process-wide and can only be configured once, so both the runtime stage and
/// the editor trainer have to come through here rather than each calling
/// <see cref="PythonEngine.Initialize()"/> with their own paths.
/// </summary>
public static class PythonRuntime
{
    /// <summary>
    /// Environment variable naming the CPython shared library, checked before the serialized path.
    /// </summary>
    /// <remarks>
    /// An interpreter lives wherever a particular machine put it, so the path is a property of the
    /// machine, not of the project. Serializing it into an asset checks one developer's layout into
    /// the repository and breaks it for everyone else; these two variables are how a second machine
    /// works without editing a shared asset. The serialized paths are kept as the fallback so an
    /// existing setup keeps working untouched.
    /// </remarks>
    public const string PythonDllVariable = "MOSYNTH_PYTHON_DLL";

    /// <summary>Environment variable naming the virtual environment. See <see cref="PythonDllVariable"/>.</summary>
    public const string PythonVenvVariable = "MOSYNTH_PYTHON_VENV";

    private static bool _pythonDllAssigned;
    private static string _scriptsFolder;

    /// <summary>The repository's Python folder, which is where our modules are imported from.</summary>
    public static string ScriptsFolder =>
        _scriptsFolder ??= Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Python"));

    /// <summary>
    /// The CPython shared library that will actually be used: <see cref="PythonDllVariable"/> if it
    /// is set, otherwise <paramref name="configuredPath"/>. Empty means neither is set, and
    /// pythonnet falls back to PYTHONNET_PYDLL on its own.
    /// </summary>
    public static string ResolvePythonDll(string configuredPath) =>
        Resolve(PythonDllVariable, configuredPath);

    /// <summary>
    /// The virtual environment that will actually be used: <see cref="PythonVenvVariable"/> if it is
    /// set, otherwise <paramref name="configuredPath"/>. Empty means the interpreter's own
    /// site-packages.
    /// </summary>
    public static string ResolveVenv(string configuredPath) =>
        Resolve(PythonVenvVariable, configuredPath);

    private static string Resolve(string variable, string configuredPath)
    {
        var fromEnvironment = Environment.GetEnvironmentVariable(variable);
        return string.IsNullOrWhiteSpace(fromEnvironment) ? configuredPath ?? "" : fromEnvironment;
    }

    /// <summary>
    /// Start CPython if it is not already running and make the project's Python modules importable.
    /// Safe to call repeatedly.
    /// </summary>
    /// <param name="pythonDllPath">
    /// Full path to the CPython shared library, or empty. Only meaningful on the very first call of
    /// the process -- pythonnet throws if it is changed after the interpreter starts.
    /// <see cref="PythonDllVariable"/> overrides it, and with neither set pythonnet falls back to
    /// the PYTHONNET_PYDLL environment variable.
    /// </param>
    /// <param name="venvPath">
    /// Virtual environment supplying numpy / scipy / torch, or empty.
    /// <see cref="PythonVenvVariable"/> overrides it.
    /// </param>
    public static void EnsureInitialized(string pythonDllPath, string venvPath)
    {
        pythonDllPath = ResolvePythonDll(pythonDllPath);
        venvPath = ResolveVenv(venvPath);

        if (!PythonEngine.IsInitialized)
        {
            if (!_pythonDllAssigned && !string.IsNullOrWhiteSpace(pythonDllPath))
            {
                if (!File.Exists(pythonDllPath))
                {
                    throw new FileNotFoundException(
                        $"Python DLL not found at '{pythonDllPath}'. Set {PythonDllVariable} for this " +
                        "machine, or the path on the MotionFieldConfig.",
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
                string sitePackages = Path.Combine(venvPath, "Lib", "site-packages");
                if (!Directory.Exists(sitePackages))
                {
                    throw new DirectoryNotFoundException(
                        $"No site-packages under '{venvPath}'. Set {PythonVenvVariable} for this " +
                        "machine, or the venv path on the MotionFieldConfig.");
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
    /// <param name="moduleName"></param>
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
    /// Forget every already-imported module that lives under <see cref="ScriptsFolder"/>, so the
    /// next import re-reads them from disk. Returns how many were dropped.
    /// </summary>
    /// <remarks>
    /// This replaces a per-module <c>importlib.reload</c>, which only re-executes the one module it
    /// is handed. Its dependencies stay cached, so reloading <c>MotionField</c> after adding a
    /// symbol to <c>motion_field_io</c> re-runs the new import line against the old dependency and
    /// dies on ImportError -- editing a shared module was effectively impossible without restarting
    /// the editor.
    ///
    /// Dropping the graph wholesale also means a group of imports issued after one call sees a
    /// single consistent generation of the code. Reloading module by module does not: each reload
    /// rebinds only that module's classes, so two modules can end up holding different class
    /// objects for the same class.
    ///
    /// Live Python objects created before the call keep working -- they hold a reference to the
    /// class they were built from, which simply is no longer the one a fresh import returns. Call
    /// this only where everything downstream is about to be rebuilt.
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
}
}
