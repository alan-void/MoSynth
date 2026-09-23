using System.Collections.Generic;
using Python.Runtime;

namespace AnimationTools
{
/// <summary>
/// Marshals a "which bones does this model predict" selection across the PythonNET boundary, and
/// resolves the same selection against a live skeleton on the C# side.
/// </summary>
/// <remarks>
/// Shared by every synthesis method so training and runtime derive the bone set identically.
/// A selection is authored as excluded bone names, and excluding a bone excludes its whole
/// <see cref="Subtree"/>, because a predicted rotation needs its parent's frame.
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
    /// A missing name or a mismatched order means the model would be fed something other than what
    /// it was trained on, so both fail loudly rather than being worked around.
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
