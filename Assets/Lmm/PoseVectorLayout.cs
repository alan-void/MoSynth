namespace Lmm
{
/// <summary>
/// Where each block of a decompressed pose vector starts, read off the checkpoint at load.
/// </summary>
/// <remarks>
/// The layout is Python's — <c>lmm.dataset.pose_vector_layout</c> — and this is the C# side reading
/// it back rather than declaring it a second time. That matters because blocks are appended over
/// time and every earlier block of an older checkpoint still slices out correctly, so a width alone
/// cannot tell a stale file from a current one. <see cref="TryBind"/> checks the names, in order,
/// and refuses anything it does not recognise instead of reading a new block off an old offset.
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
    /// The root bone's height above the ground plane, one float. Its other two coordinates are
    /// what the character frame transform removed, so this is all of its position that is free.
    /// </summary>
    public readonly int RootHeight;

    /// <summary>The character frame's own travel, three floats, metres per second.</summary>
    public readonly int RootVelocity;

    /// <summary>The character frame's own turn, one float, radians per second.</summary>
    public readonly int RootYawRate;

    /// <summary>Left then right foot contact, two floats read as flags above 0.5.</summary>
    public readonly int Contacts;

    /// <summary>Total floats in a pose vector.</summary>
    public readonly int FloatCount;

    private PoseVectorLayout(int rotations, int velocities, int angularVelocities,
        int rootHeight, int rootVelocity, int rootYawRate, int contacts, int floatCount)
    {
        Rotations = rotations;
        Velocities = velocities;
        AngularVelocities = angularVelocities;
        RootHeight = rootHeight;
        RootVelocity = rootVelocity;
        RootYawRate = rootYawRate;
        Contacts = contacts;
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

    /// <summary>Floats in each block that is not per-bone, or 0 for one that is.</summary>
    private static readonly int[] FixedFloats = { 0, 0, 0, 1, 3, 1, 2 };

    /// <summary>
    /// Reads a checkpoint's declared pose layout, refusing one this code cannot interpret.
    /// </summary>
    /// <param name="names">Block names in packing order, from <c>LmmPolicy.pose_blocks</c>.</param>
    /// <param name="offsetsAndCounts">
    /// Offset and float count per block, flattened in the same order, from
    /// <c>LmmPolicy.pose_block_offsets</c>.
    /// </param>
    /// <param name="boneCount">How many bones the checkpoint predicts.</param>
    /// <param name="error">Why the layout was refused, suitable for a console message.</param>
    public static bool TryBind(string[] names, int[] offsetsAndCounts, int boneCount,
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

            var expectedCount = FloatsPerBone[i] * boneCount + FixedFloats[i];
            var count = offsetsAndCounts[i * 2 + 1];
            if (count != expectedCount)
            {
                error = $"the '{names[i]}' block is {count} floats where {boneCount} bones make it " +
                        $"{expectedCount}";
                return false;
            }

            offsets[i] = offsetsAndCounts[i * 2];
        }

        var last = ExpectedNames.Length - 1;
        layout = new PoseVectorLayout(offsets[0], offsets[1], offsets[2], offsets[3],
            offsets[4], offsets[5], offsets[6],
            offsets[last] + offsetsAndCounts[last * 2 + 1]);
        error = null;
        return true;
    }
}
}
