using System;

namespace AnimationTools.Editor
{
/// <summary>
/// The handful of facts a sweep needs that cannot survive entering play mode in a static field.
/// Written to <see cref="PlanPath"/> before the play-mode domain reload and read back after it.
/// </summary>
/// <remarks>
/// Everything else the sweep needs is reachable from <see cref="configAssetPath"/>, because asset
/// references resolve fine on the far side of a reload — only the live object graph does not.
/// <c>Temp/</c> is the right home: Unity clears it between sessions, so a sweep that was killed
/// mid-run cannot resurrect itself the next time someone presses Play.
/// </remarks>
[Serializable]
public class SynthesisBenchmarkPlan
{
    public const string PlanPath = "Temp/mosynth_benchmark_plan.json";

    public string configAssetPath;

    /// <summary>Absolute path of this sweep's output folder.</summary>
    public string outputDirectory;

    /// <summary>Quit the Editor with a status code when the sweep ends. Set for headless runs.</summary>
    public bool exitEditorWhenDone;
}
}
