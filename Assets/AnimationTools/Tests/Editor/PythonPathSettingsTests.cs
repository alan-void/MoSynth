using System;
using NUnit.Framework;

namespace AnimationTools.Tests
{
public class PythonPathSettingsTests
{
    [Test]
    public void JsonRoundTripsBothPaths()
    {
        var paths = new PythonPaths
        {
            pythonDllPath = @"C:\Python\python313.dll",
            pythonVenvPath = @"D:\work\.anim_env"
        };

        var read = PythonPathSettings.Parse(PythonPathSettings.ToJson(paths));

        Assert.AreEqual(paths.pythonDllPath, read.pythonDllPath);
        Assert.AreEqual(paths.pythonVenvPath, read.pythonVenvPath);
    }

    [Test]
    public void ParseTreatsUnreadableJsonAsUnset()
    {
        foreach (var json in new[] { "", "   ", "not json at all", "{", "[1, 2, 3]" })
        {
            var read = PythonPathSettings.Parse(json);
            Assert.AreEqual("", read.pythonDllPath, $"dll path from \"{json}\"");
            Assert.AreEqual("", read.pythonVenvPath, $"venv path from \"{json}\"");
        }
    }

    [Test]
    public void ParseFillsInAMissingKey()
    {
        var read = PythonPathSettings.Parse("{\"pythonVenvPath\": \"D:/env\"}");

        Assert.AreEqual("", read.pythonDllPath);
        Assert.AreEqual("D:/env", read.pythonVenvPath);
    }
}

public class PythonRuntimeResolveTests
{
    private const string Variable = "MOSYNTH_TEST_PYTHON_PATH";

    [TearDown]
    public void ClearVariable() => Environment.SetEnvironmentVariable(Variable, null);

    [Test]
    public void SettingsPathIsUsedWhenNoVariableIsSet()
    {
        var resolved = PythonRuntime.Resolve(Variable, "from-settings", out var source);

        Assert.AreEqual("from-settings", resolved);
        Assert.AreEqual(PythonRuntime.PathSource.Settings, source);
    }

    [Test]
    public void VariableOverridesTheSettingsPath()
    {
        Environment.SetEnvironmentVariable(Variable, "from-environment");

        var resolved = PythonRuntime.Resolve(Variable, "from-settings", out var source);

        Assert.AreEqual("from-environment", resolved);
        Assert.AreEqual(PythonRuntime.PathSource.Environment, source);
    }

    [Test]
    public void ABlankVariableDoesNotCountAsSet()
    {
        Environment.SetEnvironmentVariable(Variable, "   ");

        var resolved = PythonRuntime.Resolve(Variable, "from-settings", out var source);

        Assert.AreEqual("from-settings", resolved);
        Assert.AreEqual(PythonRuntime.PathSource.Settings, source);
    }

    [Test]
    public void NeitherSourceResolvesToEmptyRatherThanNull()
    {
        var resolved = PythonRuntime.Resolve(Variable, null, out var source);

        Assert.AreEqual("", resolved);
        Assert.AreEqual(PythonRuntime.PathSource.Unset, source);
    }
}
}
