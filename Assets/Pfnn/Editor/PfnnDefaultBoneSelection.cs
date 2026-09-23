using AnimationTools;
using AnimationTools.Editor;

namespace Pfnn.Editor
{
/// <summary>
/// Writes <see cref="DefaultPredictedBones"/>' answer into a <see cref="PfnnConfig"/>.
/// </summary>
public static class PfnnDefaultBoneSelection
{
    /// <summary>
    /// Replace <paramref name="config"/>'s selection with the default for
    /// <paramref name="skeleton"/>. The caller records the undo and marks the config stale.
    /// </summary>
    public static void Apply(PfnnConfig config, Skeleton skeleton)
    {
        config.excludedBones.Clear();
        foreach (var boneName in DefaultPredictedBones.Excluded(skeleton))
        {
            config.SetPredicted(boneName, false);
        }
    }
}
}
