namespace Lmm
{
/// <summary>
/// Where each block of a decompressed pose vector starts, read off the checkpoint at load.
/// </summary>
/// <remarks>
/// The layout is declared once, by <c>lmm.dataset.pose_vector_layout</c>, and read back here.
/// <see cref="TryBind"/> checks block names in order, because a total width alone cannot tell a
/// stale checkpoint from a current one.
/// </remarks>
public readonly struct PoseVectorLayout
{
    /// <summary>
    /// Joint rotations against their parent, in the two-axis form, six floats per predicted bone.
    /// </summary>
    public readonly int Rotations;

    /// <summary>Joint linear rates, three floats per predicted bone, metres per second.</summary>
    public readonly int Velocities;

    /// <summary>Joint angular rates, three floats per predicted bone, radians per second.</summary>
    public readonly int AngularVelocities;

    /// <summary>
    /// The root bone's height above the ground plane, one float; its ground-plane coordinates are
    /// zero in the character frame.
    /// </summary>
    public readonly int RootHeight;

    /// <summary>The character frame's own travel, three floats, metres per second.</summary>
    public readonly int RootVelocity;

    /// <summary>The character frame's own turn, one float, radians per second.</summary>
    public readonly int RootYawRate;

    /// <summary>
    /// One float per contact slot, in the pipeline's slot order, read as flags above 0.5.
    /// </summary>
    public readonly int Contacts;

    /// <summary>Floats in the <see cref="Contacts"/> block: the number of contact slots.</summary>
    public readonly int ContactCount;

    /// <summary>Total floats in a pose vector.</summary>
    public readonly int FloatCount;

    private PoseVectorLayout(int rotations, int velocities, int angularVelocities,
        int rootHeight, int rootVelocity, int rootYawRate, int contacts, int contactCount, int floatCount)
    {
        Rotations = rotations;
        Velocities = velocities;
        AngularVelocities = angularVelocities;
        RootHeight = rootHeight;
        RootVelocity = rootVelocity;
        RootYawRate = rootYawRate;
        Contacts = contacts;
        ContactCount = contactCount;
        FloatCount = floatCount;
    }

    /// <summary>The blocks this code reads, in the order Python packs them.</summary>
    private static readonly string[] ExpectedNames =
    {
        "rotations_6d", "velocities", "angular_velocities",
        "root_height", "root_velocity", "root_yaw_rate", "contacts"
    };

    /// <summary>Floats per predicted bone in each block, or 0 for a block that is not per-bone.</summary>
    private static readonly int[] FloatsPerBone = { 6, 3, 3, 0, 0, 0, 0 };

    /// <summary>
    /// Floats in each block that is not per-bone, or 0 for one that is. The contact block's width is
    /// the contact count, supplied by the caller.
    /// </summary>
    private static readonly int[] FixedFloats = { 0, 0, 0, 1, 3, 1, 0 };

    private const int ContactBlock = 6;

    /// <summary>
    /// Reads a checkpoint's declared pose layout, refusing one this code cannot interpret.
    /// </summary>
    /// <param name="names">Block names in packing order, from <c>LmmPolicy.pose_blocks</c>.</param>
    /// <param name="offsetsAndCounts">
    /// Offset and float count per block, flattened in the same order, from
    /// <c>LmmPolicy.pose_block_offsets</c>.
    /// </param>
    /// <param name="boneCount">How many bones the checkpoint predicts.</param>
    /// <param name="contactCount">
    /// How many contact slots the pipeline pose has. A checkpoint predicting a different number was
    /// trained against a different contact list, and is refused.
    /// </param>
    /// <param name="error">Why the layout was refused, suitable for a console message.</param>
    public static bool TryBind(string[] names, int[] offsetsAndCounts, int boneCount, int contactCount,
        out PoseVectorLayout layout, out string error)
    {
        layout = default;

        if (names.Length != ExpectedNames.Length || offsetsAndCounts.Length != names.Length * 2)
        {
            error = $"it declares {names.Length} pose blocks where this code reads " +
                    $"{ExpectedNames.Length}";
            return false;
        }

        var offsets = new int[ExpectedNames.Length];
        for (var i = 0; i < ExpectedNames.Length; i++)
        {
            if (names[i] != ExpectedNames[i])
            {
                error = $"pose block {i} is '{names[i]}' where this code reads '{ExpectedNames[i]}'";
                return false;
            }

            var count = offsetsAndCounts[i * 2 + 1];
            if (i == ContactBlock && count != contactCount)
            {
                error = $"the '{names[i]}' block is {count} floats where this character has " +
                        $"{contactCount} contact bones";
                return false;
            }

            var expectedCount = FloatsPerBone[i] * boneCount + FixedFloats[i];
            if (i != ContactBlock && count != expectedCount)
            {
                error = $"the '{names[i]}' block is {count} floats where {boneCount} bones make it " +
                        $"{expectedCount}";
                return false;
            }

            offsets[i] = offsetsAndCounts[i * 2];
        }

        var last = ExpectedNames.Length - 1;
        layout = new PoseVectorLayout(offsets[0], offsets[1], offsets[2], offsets[3],
            offsets[4], offsets[5], offsets[ContactBlock], contactCount,
            offsets[last] + offsetsAndCounts[last * 2 + 1]);
        error = null;
        return true;
    }
}
}
