using System.Collections.Generic;
using Python.Runtime;

namespace AnimationTools
{
/// <summary>
/// Marshals a "which bones does this model predict" selection across the PythonNET boundary, and
/// resolves the same selection against a live skeleton on the C# side.
/// </summary>
/// <remarks>
/// One place, because a network trained over one bone set and run over another is exactly the
/// silent disagreement the neural work here is organised against — and a checkpoint's own bone-name
/// check can only catch it if both sides derive the set the same way. More than one synthesis
/// method now needs that, which is why this lives beside the skeleton rather than beside any one
/// of them.
/// <para>
/// A selection is authored as the bones held <em>back</em>, sparse and keyed by name, so a joint
/// that moves in the hierarchy keeps its setting. Excluding a bone excludes its whole subtree,
/// which is not a convenience: a model predicts rotations, and a rotation needs its parent's frame
/// to be applied in. <see cref="Subtree"/> is the unit an exclusion is made in.
/// </para>
/// </remarks>
public static class PredictedBoneSelection
{
    /// <summary>
    /// The excluded bone names as a Python list, for a trainer's keyword arguments. Must be called
    /// with the GIL held.
    /// </summary>
    public static PyList ToPython(IEnumerable<string> excludedBoneNames)
    {
        var list = new PyList();
        if (excludedBoneNames == null) return list;

        foreach (var boneName in excludedBoneNames)
        {
            if (string.IsNullOrEmpty(boneName)) continue;
            using var name = boneName.ToPython();
            list.Append(name);
        }

        return list;
    }

    /// <summary>
    /// Maps the bone names a checkpoint predicts onto indices into <paramref name="skeleton"/>.
    /// </summary>
    /// <remarks>
    /// This is the load-time check a checkpoint format storing names rather than indices exists to
    /// allow: a name the rig does not have, or an order that does not match, means the model is
    /// being fed something other than what it was trained on. Both are reported rather than worked
    /// around, because the alternative is a character that moves wrongly for no visible reason.
    /// </remarks>
    /// <param name="checkpointBoneNames">Bone names in the order the checkpoint packs them.</param>
    /// <param name="skeleton">The rig the stage is driving.</param>
    /// <param name="indices">One skeleton bone index per checkpoint bone, in checkpoint order.</param>
    /// <param name="error">Why the mapping failed, suitable for a console message.</param>
    public static bool TryResolve(IReadOnlyList<string> checkpointBoneNames, Skeleton skeleton,
        out int[] indices, out string error)
    {
        indices = null;
        error = null;

        if (skeleton == null || !skeleton.IsSet)
        {
            error = "the synthesis component has no skeleton";
            return false;
        }

        var resolved = new int[checkpointBoneNames.Count];
        var missing = new List<string>();
        var previous = -1;

        for (var i = 0; i < checkpointBoneNames.Count; i++)
        {
            var boneName = checkpointBoneNames[i];
            var index = skeleton.IndexOfName(boneName);
            if (index < 0)
            {
                missing.Add(boneName);
                continue;
            }

            // A checkpoint packs bones in skeleton order. Out of order here means the two skeletons
            // are not the same rig, whatever their names say.
            if (index <= previous)
            {
                error = $"bone '{boneName}' is out of order against this rig, so the checkpoint " +
                        "was trained on a different skeleton";
                return false;
            }

            previous = index;
            resolved[i] = index;
        }

        if (missing.Count > 0)
        {
            error = $"this rig has no bone named {string.Join(", ", missing)}";
            return false;
        }

        indices = resolved;
        return true;
    }

    /// <summary>
    /// Every bone strictly below <paramref name="boneIndex"/>, plus that bone. The unit a bone is
    /// excluded in — see the remarks on this class for why it cannot be one bone.
    /// </summary>
    public static List<int> Subtree(Skeleton skeleton, int boneIndex)
    {
        var subtree = new List<int> { boneIndex };

        // Bones are stored depth-first, so a descendant always follows its ancestor and its parent
        // is resolved before it is reached.
        for (var i = boneIndex + 1; i < skeleton.BoneCount; i++)
        {
            if (subtree.Contains(skeleton.GetParentIndex(i))) subtree.Add(i);
        }

        return subtree;
    }
}
}
