using Python.Runtime;

namespace MotionField
{
/// <summary>
/// Marshals <see cref="MotionFieldConfig.boneWeights"/> into the mapping Python resolves against
/// the skeleton.
/// </summary>
/// <remarks>
/// Shared by the runtime stage and the Editor's training buttons so both always send the same table.
/// </remarks>
public static class MotionFieldBoneWeights
{
    /// <summary>
    /// Build the <c>{joint name: (position, velocity)}</c> dict for
    /// <c>MotionField.resolve_bone_weights</c>. Must be called with the GIL held.
    /// </summary>
    /// <remarks>
    /// Always returns a dict, possibly empty; Python normalises an all-neutral table to uniform.
    /// </remarks>
    public static PyDict ToPython(MotionFieldConfig config)
    {
        var dict = new PyDict();
        if (config == null) return dict;

        foreach (var weight in config.boneWeights)
        {
            if (string.IsNullOrEmpty(weight.name) || weight.IsNeutral) continue;

            // The dict takes its own reference, so disposing the tuple here is safe.
            using var pair = new PyTuple(new[]
            {
                ((double)weight.position).ToPython(),
                ((double)weight.velocity).ToPython(),
            });
            dict[weight.name] = pair;
        }

        return dict;
    }
}
}
