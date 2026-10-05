using NUnit.Framework;

namespace Lmm.Tests
{
/// <summary>
/// Reading a checkpoint's declared pose layout.
/// </summary>
/// <remarks>
/// The format is unversioned, so these verify that block names and widths alone reject a layout
/// this code cannot read.
/// </remarks>
public class LmmLayoutTests
{
    private const int BoneCount = 4;
    private const int ContactCount = 2;

    /// <summary>The blocks Python packs, in order, as <c>LmmPolicy.pose_blocks</c> reports them.</summary>
    private static string[] Names() => new[]
    {
        "rotations_6d", "velocities", "angular_velocities",
        "root_height", "root_velocity", "root_yaw_rate", "contacts"
    };

    /// <summary>Offset and count per block, flattened, as <c>LmmPolicy.pose_block_offsets</c> does.</summary>
    private static int[] OffsetsAndCounts(int bones = BoneCount, int contacts = ContactCount)
    {
        int[] counts = { 6 * bones, 3 * bones, 3 * bones, 1, 3, 1, contacts };
        var flat = new int[counts.Length * 2];
        var offset = 0;

        for (var i = 0; i < counts.Length; i++)
        {
            flat[i * 2] = offset;
            flat[i * 2 + 1] = counts[i];
            offset += counts[i];
        }

        return flat;
    }

    [Test]
    public void TryBind_ReadsEveryBlockOffset()
    {
        Assert.IsTrue(PoseVectorLayout.TryBind(Names(), OffsetsAndCounts(), BoneCount, ContactCount,
            out var layout, out var error), error);

        Assert.AreEqual(0, layout.Rotations);
        Assert.AreEqual(6 * BoneCount, layout.Velocities);
        Assert.AreEqual(9 * BoneCount, layout.AngularVelocities);
        Assert.AreEqual(12 * BoneCount, layout.RootHeight);
        Assert.AreEqual(12 * BoneCount + 1, layout.RootVelocity);
        Assert.AreEqual(12 * BoneCount + 4, layout.RootYawRate);
        Assert.AreEqual(12 * BoneCount + 5, layout.Contacts);
    }

    [Test]
    public void TryBind_TotalIsTwelveFloatsPerBonePlusTheRoots()
    {
        PoseVectorLayout.TryBind(Names(), OffsetsAndCounts(), BoneCount, ContactCount, out var layout, out _);

        Assert.AreEqual(12 * BoneCount + 7, layout.FloatCount);
    }

    [Test]
    public void TryBind_RefusesARenamedBlockAndNamesIt()
    {
        var names = Names();
        names[1] = "joint_velocities";

        Assert.IsFalse(PoseVectorLayout.TryBind(names, OffsetsAndCounts(), BoneCount, ContactCount, out _,
            out var error));
        StringAssert.Contains("joint_velocities", error);
        StringAssert.Contains("velocities", error);
    }

    [Test]
    public void TryBind_RefusesAReorderedLayoutRatherThanReadingOffTheWrongOffsets()
    {
        var names = Names();
        (names[0], names[1]) = (names[1], names[0]);

        Assert.IsFalse(PoseVectorLayout.TryBind(names, OffsetsAndCounts(), BoneCount, ContactCount, out _, out _));
    }

    [Test]
    public void TryBind_RefusesABlockSizedForADifferentBoneCount()
    {
        // The checkpoint was written for five bones; this stage resolved four against the rig.
        Assert.IsFalse(PoseVectorLayout.TryBind(Names(), OffsetsAndCounts(bones: 5), BoneCount, ContactCount,
            out _, out var error));
        StringAssert.Contains("rotations_6d", error);
    }

    [Test]
    public void TryBind_RefusesAnAppendedBlockThisCodeDoesNotRead()
    {
        var names = new string[Names().Length + 1];
        Names().CopyTo(names, 0);
        names[^1] = "phase";

        var counts = OffsetsAndCounts();
        var extended = new int[counts.Length + 2];
        counts.CopyTo(extended, 0);
        extended[^2] = 12 * BoneCount + 7;
        extended[^1] = 1;

        Assert.IsFalse(PoseVectorLayout.TryBind(names, extended, BoneCount, ContactCount, out _, out var error));
        StringAssert.Contains("pose blocks", error);
    }

    [Test]
    public void TryBind_ReadsAContactBlockAsWideAsTheContactList()
    {
        Assert.IsTrue(PoseVectorLayout.TryBind(Names(), OffsetsAndCounts(contacts: 3), BoneCount, 3,
            out var layout, out var error), error);

        Assert.AreEqual(3, layout.ContactCount);
        Assert.AreEqual(12 * BoneCount + 5, layout.Contacts);
        Assert.AreEqual(12 * BoneCount + 8, layout.FloatCount);
    }

    [Test]
    public void TryBind_RefusesAContactBlockSizedForADifferentContactList()
    {
        Assert.IsFalse(PoseVectorLayout.TryBind(Names(), OffsetsAndCounts(contacts: 2), BoneCount, 3,
            out _, out var error));
        StringAssert.Contains("contacts", error);
    }
}
}
