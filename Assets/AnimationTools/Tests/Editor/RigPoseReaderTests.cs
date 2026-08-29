using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// Reading a live rig into a pose buffer. The rates are the whole point: a pose carries per-second
/// rates with angular rates as rotation vectors in radians, and everything downstream —
/// <see cref="SkeletonData.CharacterSpaceVelocity"/> included — reads them as such.
/// </summary>
public class RigPoseReaderTests
{
    private const float DeltaTime = 1f / 30f;

    private Skeleton _skeleton;
    private Transform[] _transforms;
    private PoseBuffer _pose;

    [SetUp]
    public void SetUp()
    {
        _skeleton = TestSkeletons.CreateChain3();
        _transforms = new Transform[_skeleton.BoneCount];
        for (var i = 0; i < _transforms.Length; i++)
        {
            _transforms[i] = _skeleton.GetBone(i).Transform;
        }

        _pose = PoseBuffer.Allocate(PoseLayoutBuilder.Build(_skeleton, out _), Allocator.Temp);
        RigPoseReader.Seed(_pose, _transforms);
    }

    [TearDown]
    public void TearDown()
    {
        _pose.Dispose();
        TestSkeletons.DestroyAll();
    }

    [Test]
    public void Seed_LeavesEveryRateAtZero()
    {
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            Assert.That(math.length(_pose.Velocities[i]), Is.EqualTo(0f).Within(1e-6f));
            Assert.That(math.length(_pose.AngularVelocities[i]), Is.EqualTo(0f).Within(1e-6f));
        }
    }

    [Test]
    public void Read_ReportsLinearRatesPerSecond()
    {
        _transforms[0].position += new Vector3(0.05f, 0f, 0f);

        RigPoseReader.Read(_pose, _transforms, DeltaTime);

        Assert.That(_pose.Velocities[0].x, Is.EqualTo(0.05f / DeltaTime).Within(1e-3f));
    }

    [Test]
    public void Read_ReportsAngularRatesAsRadiansPerSecond()
    {
        // A quarter turn about +Y over one tick, i.e. (pi/2) / dt rad/s about +Y.
        _transforms[1].localRotation = Quaternion.Euler(0f, 90f, 0f);

        RigPoseReader.Read(_pose, _transforms, DeltaTime);

        var rate = _pose.AngularVelocities[1];
        Assert.That(rate.y, Is.EqualTo(0.5f * math.PI / DeltaTime).Within(1e-2f),
            "Degrees, or a per-tick delta, would land nowhere near this.");
        Assert.That(math.abs(rate.x), Is.LessThan(1e-3f));
        Assert.That(math.abs(rate.z), Is.LessThan(1e-3f));
    }

    [Test]
    public void Read_TakesAnAngularRateTheShortWayRound()
    {
        // -10 degrees. Euler angles would report this as +350, and a signed rotation vector as -10.
        _transforms[1].localRotation = Quaternion.Euler(0f, -10f, 0f);

        RigPoseReader.Read(_pose, _transforms, DeltaTime);

        var rate = _pose.AngularVelocities[1];
        Assert.That(rate.y, Is.EqualTo(math.radians(-10f) / DeltaTime).Within(1e-2f));
        Assert.That(math.length(rate), Is.LessThan(math.PI / DeltaTime),
            "A rate above pi per tick means the long way round was taken.");
    }

    [Test]
    public void Read_WithAZeroTimestep_ReportsNoRateRatherThanInfinity()
    {
        _transforms[0].position += new Vector3(0.05f, 0f, 0f);
        _transforms[1].localRotation = Quaternion.Euler(0f, 90f, 0f);

        RigPoseReader.Read(_pose, _transforms, 0f);

        Assert.That(math.length(_pose.Velocities[0]), Is.EqualTo(0f).Within(1e-6f));
        Assert.That(math.length(_pose.AngularVelocities[1]), Is.EqualTo(0f).Within(1e-6f));
    }

    [Test]
    public void Read_TakesBoneZeroInWorldSpaceAndTheRestParentLocal()
    {
        _transforms[0].position = new Vector3(3f, 4f, 5f);
        _transforms[1].localRotation = Quaternion.Euler(0f, 45f, 0f);

        RigPoseReader.Read(_pose, _transforms, DeltaTime);

        Assert.That(math.distance(_pose.Positions[0], new float3(3f, 4f, 5f)), Is.LessThan(1e-5f));
        Assert.That(math.abs(math.dot(_pose.Rotations[1], (quaternion)_transforms[1].localRotation)),
            Is.EqualTo(1f).Within(1e-5f));
    }

    [Test]
    public void Read_LeavesBoneOffsetsAlone()
    {
        // Only bone 0 carries a position; the rest hold the rest offsets that fix bone lengths, and
        // rereading a moved rig must not start baking the world into them.
        var restOffset = _pose.Positions[1];
        _transforms[0].position += new Vector3(0f, 0f, 2f);

        RigPoseReader.Read(_pose, _transforms, DeltaTime);

        Assert.That(math.distance(_pose.Positions[1], restOffset), Is.LessThan(1e-5f));
    }

    [Test]
    public void Read_LeavesFootContactsToWhicheverStageOwnsThem()
    {
        // The builder caches per skeleton, so this is the layout the buffer was allocated over.
        PoseLayoutBuilder.Build(_skeleton, out var contacts);
        _pose.SetBool(contacts.Left, true);

        RigPoseReader.Read(_pose, _transforms, DeltaTime);

        Assert.IsTrue(_pose.GetBool(contacts.Left));
    }
}
}
