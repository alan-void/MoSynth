using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// Writes every test run's summary to Temp/animtools_test_results.txt so external tooling
/// can read results across the domain reloads a test run triggers.
/// </summary>
[InitializeOnLoad]
public static class TestResultDump
{
    static TestResultDump()
    {
        var api = ScriptableObject.CreateInstance<TestRunnerApi>();
        api.RegisterCallbacks(new Callbacks());
    }

    public static void Run()
    {
        var api = ScriptableObject.CreateInstance<TestRunnerApi>();
        var filter = new Filter
        {
            testMode = TestMode.EditMode,
            assemblyNames = new[] { "AnimationTools.Tests", "MotionMatching.Tests", "Pfnn.Tests", "Lmm.Tests" }
        };
        api.Execute(new ExecutionSettings(filter));
    }

    private class Callbacks : ICallbacks
    {
        private readonly StringBuilder failures = new();

        public void RunStarted(ITestAdaptor testsToRun)
        {
            failures.Clear();
        }

        public void RunFinished(ITestResultAdaptor result)
        {
            var summary = $"passed={result.PassCount} failed={result.FailCount} skipped={result.SkipCount}\n{failures}";
            File.WriteAllText(Path.Combine(Application.dataPath, "../Temp/animtools_test_results.txt"), summary);
        }

        public void TestStarted(ITestAdaptor test)
        {
        }

        public void TestFinished(ITestResultAdaptor result)
        {
            if (result.TestStatus == TestStatus.Failed && !result.HasChildren)
                failures.AppendLine($"FAIL {result.FullName}: {result.Message}");
        }
    }
}
}
