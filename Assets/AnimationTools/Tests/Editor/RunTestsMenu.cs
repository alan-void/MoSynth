using UnityEditor;

namespace AnimationTools.Tests
{
/// <summary>
/// Runs the EditMode suites from a menu item, so a headless or scripted caller can trigger a run
/// and then read <see cref="TestResultDump"/>'s summary file — the Test Runner window itself is
/// not reachable that way.
/// </summary>
public static class RunTestsMenu
{
    [MenuItem("MoSynth/Tests/Run EditMode Tests")]
    public static void RunEditModeTests() => TestResultDump.Run();
}
}
