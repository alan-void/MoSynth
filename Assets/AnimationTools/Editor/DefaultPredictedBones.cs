using System;
using System.Collections.Generic;
using System.Linq;

namespace AnimationTools.Editor
{
/// <summary>
/// The one-click starting point for a predicted-bone selection: hold back the fingers and the leaf
/// tips, predict everything else.
/// </summary>
/// <remarks>
/// A name heuristic, and deliberately confined to an authoring button rather than run anywhere in
/// a training or inference path. Fingers barely move in locomotion, so a model that predicts them
/// spends most of its capacity learning constants — but which bones those are is a property of a
/// particular rig. Pressing the button writes the answer into the asset, where it can be read,
/// corrected, and seen to be what the model was trained on.
/// </remarks>
public static class DefaultPredictedBones
{
    /// <summary>Bones whose descendants are held back. The named bone itself is kept.</summary>
    private static readonly string[] SubtreeRootsHeldBack = { "LeftHand", "RightHand" };

    /// <summary>
    /// Suffix marking a rig's leaf tips. They exist to give the bone above them a direction and
    /// carry no motion of their own.
    /// </summary>
    private const string LeafSuffix = "_end";

    /// <summary>
    /// The bone names a model over <paramref name="skeleton"/> should hold back by default, in
    /// skeleton order. Empty for a skeleton that is not set.
    /// </summary>
    /// <remarks>
    /// Returns the names rather than writing them, because the assets that carry such a list differ
    /// in how they record a change — undo, a dirty flag, a stale-checkpoint flag — and that is the
    /// caller's business.
    /// </remarks>
    public static List<string> Excluded(Skeleton skeleton)
    {
        var excluded = new List<string>();
        if (skeleton == null || !skeleton.IsSet) return excluded;

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            var boneName = skeleton.GetBone(i).Name;

            if (boneName.EndsWith(LeafSuffix, StringComparison.Ordinal))
            {
                if (!excluded.Contains(boneName)) excluded.Add(boneName);
                continue;
            }

            if (!IsSubtreeRootHeldBack(boneName)) continue;

            // The hand itself stays: it is where the arm ends, and what the fingers hang off.
            foreach (var descendant in PredictedBoneSelection.Subtree(skeleton, i).Skip(1))
            {
                var name = skeleton.GetBone(descendant).Name;
                if (!excluded.Contains(name)) excluded.Add(name);
            }
        }

        return excluded;
    }

    /// <summary>
    /// Matched on the end of the name so a rig that prefixes its bones — <c>Model:LeftHand</c> —
    /// is recognised without the prefix having to be configured.
    /// </summary>
    private static bool IsSubtreeRootHeldBack(string boneName) =>
        SubtreeRootsHeldBack.Any(root => boneName.EndsWith(root, StringComparison.Ordinal));
}
}
