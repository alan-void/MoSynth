using System.IO;
using System.Text.RegularExpressions;
using AnimationTools;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.TestTools;

namespace AnimationTools.Tests
{
/// <summary>
/// Round-trip coverage for the .mmpose format. The file carries a skeleton block ahead of the
/// poses — C# does not need it, but the Python half has no ScriptableObject to read the bone tree
/// from — and nothing in the format is versioned, so that block is also what catches a database
/// extracted over a different rig. Both halves are exercised here.
/// </summary>
public class PoseSerializerTests
{
    private string _directory;

    [SetUp]
    public void SetUp()
    {
        _directory = Path.Combine(Path.GetTempPath(), "PoseSerializerTests", Path.GetRandomFileName());
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
        TestSkeletons.DestroyAll();
    }

    /// <summary>Two clips of three frames each, every channel given a distinct value per frame
    /// and per bone so a misaligned read cannot pass by coincidence.</summary>
    private static PoseSet BuildPoseSet(Skeleton skeleton)
    {
        var poseSet = new PoseSet();
        poseSet.SetSkeleton(skeleton);

        for (var clip = 0; clip < 2; clip++)
        {
            var frames = poseSet.BeginClip(3, 1f / 30f);
            for (var f = 0; f < 3; f++)
            {
                var frame = frames[f];

                // The channel properties hand back NativeSlice by value, so they have to be held
                // in a local to be written through — the same shape Deserialize uses.
                var positions = frame.Positions;
                var rotations = frame.Rotations;
                var velocities = frame.Velocities;
                var angularVelocities = frame.AngularVelocities;

                for (var b = 0; b < skeleton.BoneCount; b++)
                {
                    var seed = clip * 100 + f * 10 + b;
                    positions[b] = new float3(seed, seed + 0.5f, seed + 0.25f);
                    rotations[b] = math.normalize(new quaternion(seed, seed + 1f, seed + 2f, seed + 3f));
                    velocities[b] = new float3(-seed, seed * 2f, seed * 3f);
                    angularVelocities[b] = new float3(seed * 4f, -seed, seed * 0.5f);
                }

                frame.SetBool(poseSet.LeftFootContactHandle, f % 2 == 0);
                frame.SetBool(poseSet.RightFootContactHandle, f % 2 == 1);
            }
        }

        return poseSet;
    }

    [Test]
    public void RoundTrip_PreservesClipsAndEveryPoseChannel()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var written = BuildPoseSet(skeleton);

        new PoseSerializer().Serialize(written, _directory, "db");

        Assert.IsTrue(new PoseSerializer().Deserialize(_directory, "db", skeleton, out var read));

        Assert.AreEqual(written.NumberClips, read.NumberClips);
        Assert.AreEqual(written.NumberPoses, read.NumberPoses);

        for (var i = 0; i < written.NumberClips; i++)
        {
            Assert.AreEqual(written.GetAnimationClip(i).Start, read.GetAnimationClip(i).Start);
            Assert.AreEqual(written.GetAnimationClip(i).End, read.GetAnimationClip(i).End);
            Assert.AreEqual(written.GetAnimationClip(i).FrameTime, read.GetAnimationClip(i).FrameTime, 1e-6f);
        }

        for (var p = 0; p < written.NumberPoses; p++)
        {
            var expected = written.GetPoseBuffer(p);
            var actual = read.GetPoseBuffer(p);

            for (var b = 0; b < skeleton.BoneCount; b++)
            {
                Assert.AreEqual(expected.Positions[b], actual.Positions[b], $"position, pose {p} bone {b}");
                Assert.AreEqual(expected.Rotations[b].value, actual.Rotations[b].value, $"rotation, pose {p} bone {b}");
                Assert.AreEqual(expected.Velocities[b], actual.Velocities[b], $"velocity, pose {p} bone {b}");
                Assert.AreEqual(expected.AngularVelocities[b], actual.AngularVelocities[b],
                    $"angular velocity, pose {p} bone {b}");
            }

            Assert.AreEqual(expected.GetBool(written.LeftFootContactHandle),
                actual.GetBool(read.LeftFootContactHandle), $"left contact, pose {p}");
            Assert.AreEqual(expected.GetBool(written.RightFootContactHandle),
                actual.GetBool(read.RightFootContactHandle), $"right contact, pose {p}");
        }

        written.Dispose();
        read.Dispose();
    }

    [Test]
    public void Deserialize_RejectsDatabaseWithDifferentBoneCount()
    {
        var written = BuildPoseSet(TestSkeletons.CreateChain3());
        new PoseSerializer().Serialize(written, _directory, "db");
        written.Dispose();

        LogAssert.Expect(LogType.Error, new Regex("skeleton of 3 bones"));

        Assert.IsFalse(new PoseSerializer()
            .Deserialize(_directory, "db", TestSkeletons.CreateBranch4(), out _));
    }

    /// <summary>
    /// The case a bone count alone would miss: same number of bones, different tree. This is what
    /// stands in for the format version the file no longer carries.
    /// </summary>
    [Test]
    public void Deserialize_RejectsDatabaseWithSameCountButDifferentBones()
    {
        var written = BuildPoseSet(TestSkeletons.CreateChain3());
        new PoseSerializer().Serialize(written, _directory, "db");
        written.Dispose();

        var renamed = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("root", -1, float3.zero),
            new TestSkeletons.BoneSpec("spine", 0, new float3(0f, 1f, 0f)),
            new TestSkeletons.BoneSpec("neck", 1, new float3(0f, 1f, 0f)));

        LogAssert.Expect(LogType.Error, new Regex("\"head\" at bone 2"));

        Assert.IsFalse(new PoseSerializer().Deserialize(_directory, "db", renamed, out _));
    }

    [Test]
    public void Deserialize_RejectsDatabaseWithSameNamesButDifferentParents()
    {
        var written = BuildPoseSet(TestSkeletons.CreateChain3());
        new PoseSerializer().Serialize(written, _directory, "db");
        written.Dispose();

        // Same three names, same DFS order, but head hangs off root instead of spine.
        var reparented = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("root", -1, float3.zero),
            new TestSkeletons.BoneSpec("spine", 0, new float3(0f, 1f, 0f)),
            new TestSkeletons.BoneSpec("head", 0, new float3(0f, 2f, 0f)));

        LogAssert.Expect(LogType.Error, new Regex("parents bone 2"));

        Assert.IsFalse(new PoseSerializer().Deserialize(_directory, "db", reparented, out _));
    }

    /// <summary>
    /// An interrupted write leaves a file whose header promises more poses than it holds. The
    /// skeleton block still matches, so nothing upstream catches it and the read itself has to.
    /// </summary>
    [Test]
    public void Deserialize_RejectsATruncatedFile()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var written = BuildPoseSet(skeleton);
        new PoseSerializer().Serialize(written, _directory, "db");
        written.Dispose();

        var path = Path.Combine(_directory, "db.mmpose");
        var bytes = File.ReadAllBytes(path);
        File.WriteAllBytes(path, bytes[..(bytes.Length - 40)]);

        LogAssert.Expect(LogType.Error, new Regex("truncated"));

        Assert.IsFalse(new PoseSerializer().Deserialize(_directory, "db", skeleton, out _));
    }
}
}
