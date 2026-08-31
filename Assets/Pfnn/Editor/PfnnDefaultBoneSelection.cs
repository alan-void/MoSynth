using System;
using System.Linq;
using AnimationTools;

namespace Pfnn.Editor
{
/// <summary>
/// The one-click starting point for <see cref="PfnnConfig.excludedBones"/>: hold back the fingers
/// and the leaf tips, predict everything else.
/// </summary>
/// <remarks>
/// A name heuristic, and deliberately confined to an authoring button rather than run anywhere in
/// the training or inference path. Fingers barely move in locomotion, so a model that predicts them
/// spends most of its capacity learning constants — but which bones those are is a property of a
/// particular rig. Pressing the button writes the answer into the asset, where it can be read,
/// corrected, and seen to be what the model was trained on.
/// </remarks>
public static class PfnnDefaultBoneSelection
{
    /// <summary>Bones whose descendants are held back. The named bone itself is kept.</summary>
    private static readonly string[] SubtreeRootsHeldBack = { "LeftHand", "RightHand" };

    /// <summary>
    /// Suffix marking a rig's leaf tips. They exist to give the bone above them a direction and
    /// carry no motion of their own.
    /// </summary>
    private const string LeafSuffix = "_end";

    /// <summary>
    /// Replace <paramref name="config"/>'s selection with the default for
    /// <paramref name="skeleton"/>. The caller records the undo and marks the config stale.
    /// </summary>
    public static void Apply(PfnnConfig config, Skeleton skeleton)
    {
        config.excludedBones.Clear();
        if (skeleton == null || !skeleton.IsSet) return;

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            var boneName = skeleton.GetBone(i).Name;

            if (boneName.EndsWith(LeafSuffix, StringComparison.Ordinal))
            {
                config.SetPredicted(boneName, false);
                continue;
            }

            if (!IsSubtreeRootHeldBack(boneName)) continue;

            // The hand itself stays: it is where the arm ends, and what the fingers hang off.
            foreach (var descendant in PfnnBoneSelection.Subtree(skeleton, i).Skip(1))
            {
                config.SetPredicted(skeleton.GetBone(descendant).Name, false);
            }
        }
    }

    /// <summary>
    /// Matched on the end of the name so a rig that prefixes its bones — <c>Model:LeftHand</c> —
    /// is recognised without the prefix having to be configured.
    /// </summary>
    private static bool IsSubtreeRootHeldBack(string boneName) =>
        SubtreeRootsHeldBack.Any(root => boneName.EndsWith(root, StringComparison.Ordinal));
}
}
