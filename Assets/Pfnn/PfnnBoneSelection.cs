using System.Collections.Generic;
using AnimationTools;
using Python.Runtime;

namespace Pfnn
{
/// <summary>
/// <see cref="PfnnConfig"/>'s view of <see cref="PredictedBoneSelection"/>: the same marshalling
/// and resolution, taking the selection off the config rather than as a bare list.
/// </summary>
/// <remarks>
/// The logic moved to <c>AnimationTools</c> when a second synthesis method needed it. Keeping this
/// as the config-shaped entry point means a caller that has a <see cref="PfnnConfig"/> does not
/// have to know where the selection is stored on it.
/// </remarks>
public static class PfnnBoneSelection
{
    /// <summary>
    /// The excluded bone names as a Python list, for <c>pfnn.trainer.train</c>. Must be called with
    /// the GIL held.
    /// </summary>
    public static PyList ToPython(PfnnConfig config) =>
        PredictedBoneSelection.ToPython(config == null ? null : config.excludedBones);

    /// <inheritdoc cref="PredictedBoneSelection.TryResolve"/>
    public static bool TryResolve(IReadOnlyList<string> checkpointBoneNames, Skeleton skeleton,
        out int[] indices, out string error) =>
        PredictedBoneSelection.TryResolve(checkpointBoneNames, skeleton, out indices, out error);

    /// <inheritdoc cref="PredictedBoneSelection.Subtree"/>
    public static List<int> Subtree(Skeleton skeleton, int boneIndex) =>
        PredictedBoneSelection.Subtree(skeleton, boneIndex);
}
}
