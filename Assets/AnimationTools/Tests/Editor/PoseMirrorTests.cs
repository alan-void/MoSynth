using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// Mirroring a pose left-to-right on a rig whose two sides use different local-axis conventions.
/// </summary>
/// <remarks>
/// The right side's bones are rolled 180° about their own axis at rest, with offsets written so the
/// rest geometry is still symmetric. A mirror that reflected local rotations directly would pass on
/// a rig authored with matching axes and fail here.
/// </remarks>
public class PoseMirrorTests
{
    private const float Tolerance = 1e-4f;

    private const int Hips = 0;
    private const int Spine = 1;
    private const int LeftHand = 2;
    private const int RightHand = 3;
    private const int LeftUpLeg = 4;
    private const int LeftLeg = 5;
    private const int LeftFoot = 6;
    private const int RightUpLeg = 7;
    private const int RightLeg = 8;
    private const int RightFoot = 9;

    private GameObject _rig;
    private Skeleton _skeleton;
    private PoseMirror _mirror;
    private PoseLayout _layout;

    [SetUp]
    public void SetUp()
    {
        var roll = quaternion.RotateY(math.PI);
        var handRoll = quaternion.RotateX(math.PI / 2f);

        var hips = AddBone("mixamorig:Hips", null, new float3(0f, 1f, 0f), quaternion.identity);
        var spine = AddBone("mixamorig:Spine", hips, new float3(0f, 0.3f, 0f), quaternion.identity);
        AddBone("Hand.L", spine, new float3(-0.4f, 0.2f, 0f), quaternion.identity);
        // Rolled about x, so its world offset from spine is still +x but its axes differ.
        AddBone("Hand.R", spine, new float3(0.4f, 0.2f, 0f), handRoll);

        var leftUpLeg = AddBone("mixamorig:LeftUpLeg", hips, new float3(-0.1f, 0f, 0f), quaternion.identity);
        var leftLeg = AddBone("mixamorig:LeftLeg", leftUpLeg, new float3(0f, -0.5f, 0f), quaternion.identity);
        AddBone("mixamorig:LeftFoot", leftLeg, new float3(0f, -0.5f, 0.1f), quaternion.identity);

        // Rolled 180° about the leg's own (y) axis, so a forward offset below it is written as -z.
        var rightUpLeg = AddBone("mixamorig:RightUpLeg", hips, new float3(0.1f, 0f, 0f), roll);
        var rightLeg = AddBone("mixamorig:RightLeg", rightUpLeg, new float3(0f, -0.5f, 0f), quaternion.identity);
        AddBone("mixamorig:RightFoot", rightLeg, new float3(0f, -0.5f, -0.1f), quaternion.identity);

        _rig = hips.gameObject;
        _skeleton = new Skeleton(hips);
        Assert.IsTrue(PoseMirror.TryCreate(_skeleton, out _mirror, out var error), error);
        _layout = PoseLayoutBuilder.Build(_skeleton, out _);
    }

    [TearDown]
    public void TearDown()
    {
        if (_rig != null) Object.DestroyImmediate(_rig);
        Skeleton.InvalidateAll();
    }

    private static Transform AddBone(string name, Transform parent, float3 localPosition, quaternion localRotation)
    {
        var transform = new GameObject(name).transform;
        transform.SetParent(parent, false);
        transform.localPosition = localPosition;
        transform.localRotation = localRotation;
        return transform;
    }

    [Test]
    public void CounterpartsPairBySideTokenAndSuffix()
    {
        Assert.AreEqual(Hips, _mirror.Counterpart(Hips));
        Assert.AreEqual(Spine, _mirror.Counterpart(Spine));
        Assert.AreEqual(RightHand, _mirror.Counterpart(LeftHand));
        Assert.AreEqual(LeftHand, _mirror.Counterpart(RightHand));
        Assert.AreEqual(RightUpLeg, _mirror.Counterpart(LeftUpLeg));
        Assert.AreEqual(LeftLeg, _mirror.Counterpart(RightLeg));
        Assert.AreEqual(RightFoot, _mirror.Counterpart(LeftFoot));
        Assert.AreEqual(LeftFoot, _mirror.Counterpart(RightFoot));
    }

    [Test]
    public void PlaneNormalPointsFromRightToLeft()
    {
        AssertClose(new float3(-1f, 0f, 0f), _mirror.PlaneNormal, "plane normal");
    }

    [Test]
    public void MirroredWorldPositionsAreReflectedCounterparts()
    {
        var pose = PoseBuffer.Allocate(_layout, Allocator.Temp);
        var mirrored = PoseBuffer.Allocate(_layout, Allocator.Temp);
        try
        {
            FillPosed(pose);
            _mirror.Mirror(pose, mirrored);

            var original = WorldPositions(pose);
            var reflected = WorldPositions(mirrored);
            var normal = _mirror.PlaneNormal;
            for (var bone = 0; bone < _skeleton.BoneCount; bone++)
            {
                var source = original[_mirror.Counterpart(bone)];
                var expected = source - 2f * math.dot(source, normal) * normal;
                AssertClose(expected, reflected[bone], $"bone {bone}");
            }

            original.Dispose();
            reflected.Dispose();
        }
        finally
        {
            pose.Dispose();
            mirrored.Dispose();
        }
    }

    [Test]
    public void MirroringTwiceRestoresThePose()
    {
        var pose = PoseBuffer.Allocate(_layout, Allocator.Temp);
        var twice = PoseBuffer.Allocate(_layout, Allocator.Temp);
        try
        {
            FillPosed(pose);
            _mirror.Mirror(pose, twice);
            _mirror.Mirror(twice, twice);

            AssertSameLocalPose(pose, twice);
        }
        finally
        {
            pose.Dispose();
            twice.Dispose();
        }
    }

    [Test]
    public void RestPoseMirrorsToItself()
    {
        var pose = PoseBuffer.Allocate(_layout, Allocator.Temp);
        var mirrored = PoseBuffer.Allocate(_layout, Allocator.Temp);
        try
        {
            var positions = pose.Positions;
            var rotations = pose.Rotations;
            for (var bone = 0; bone < _skeleton.BoneCount; bone++)
            {
                positions[bone] = _skeleton.GetBone(bone).RestLocalPosition;
                rotations[bone] = _skeleton.GetBone(bone).RestLocalRotation;
            }

            _mirror.Mirror(pose, mirrored);

            AssertSameLocalPose(pose, mirrored);
        }
        finally
        {
            pose.Dispose();
            mirrored.Dispose();
        }
    }

    /// <summary>Rest offsets with a distinct, non-trivial rotation on every bone and a moved root.</summary>
    private void FillPosed(PoseBuffer pose)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;
        for (var bone = 0; bone < _skeleton.BoneCount; bone++)
        {
            var rest = _skeleton.GetBone(bone);
            var twist = quaternion.EulerXYZ(0.3f + 0.1f * bone, -0.2f * bone, 0.15f + 0.05f * bone);
            positions[bone] = rest.RestLocalPosition;
            rotations[bone] = math.mul(rest.RestLocalRotation, twist);
        }

        positions[Hips] = new float3(0.7f, 0.95f, -1.3f);
    }

    private NativeArray<float3> WorldPositions(PoseBuffer pose)
    {
        var positions = new NativeArray<float3>(_skeleton.BoneCount, Allocator.Temp);
        var rotations = new NativeArray<quaternion>(_skeleton.BoneCount, Allocator.Temp);
        _skeleton.GetSkeletonData().LocalSpaceToCharacterSpace(pose, positions, rotations);
        rotations.Dispose();
        return positions;
    }

    private void AssertSameLocalPose(PoseBuffer expected, PoseBuffer actual)
    {
        for (var bone = 0; bone < _skeleton.BoneCount; bone++)
        {
            AssertClose(expected.Positions[bone], actual.Positions[bone], $"bone {bone} position");

            // q and -q are the same rotation.
            var alignment = math.abs(math.dot(expected.Rotations[bone].value, actual.Rotations[bone].value));
            Assert.AreEqual(1f, alignment, Tolerance, $"bone {bone} rotation");
        }
    }

    private static void AssertClose(float3 expected, float3 actual, string label)
    {
        Assert.Less(math.distance(expected, actual), Tolerance, $"{label}: expected {expected}, got {actual}");
    }
}
}
