using System;
using System.Text.RegularExpressions;
using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Reflects a pose left-to-right across the skeleton's own plane of symmetry, swapping each sided
/// bone with its counterpart. Built once per bake; <see cref="Mirror"/> reuses its scratch arrays.
/// </summary>
/// <remarks>
/// Rotations are mirrored as world-space deltas from rest rather than directly, so a rig whose left
/// and right bones carry different local-axis conventions still mirrors correctly, as long as its
/// rest geometry is symmetric.
/// </remarks>
public sealed class PoseMirror
{
    private static readonly Regex SideToken = new("Left|Right|left|right");
    private static readonly Regex SideSuffix = new(@"[._][LR]$");

    private readonly int[] _parents;
    private readonly int[] _counterparts;
    private readonly quaternion[] _restWorldRotations;
    private readonly quaternion[] _restWorldRotationsInverse;
    private readonly float3 _normal;

    private readonly float3[] _worldPositions;
    private readonly quaternion[] _worldRotations;
    private readonly float3[] _mirroredPositions;
    private readonly quaternion[] _mirroredRotations;

    /// <summary>Unit normal of the mirror plane, which passes through the origin.</summary>
    public float3 PlaneNormal => _normal;

    /// <summary>The bone <paramref name="bone"/> takes its motion from; itself for an unsided bone.</summary>
    public int Counterpart(int bone) => _counterparts[bone];

    private PoseMirror(int[] parents, int[] counterparts, quaternion[] restWorldRotations, float3 normal)
    {
        var boneCount = parents.Length;
        _parents = parents;
        _counterparts = counterparts;
        _restWorldRotations = restWorldRotations;
        _restWorldRotationsInverse = new quaternion[boneCount];
        for (var i = 0; i < boneCount; i++)
        {
            _restWorldRotationsInverse[i] = math.inverse(restWorldRotations[i]);
        }

        _normal = normal;

        _worldPositions = new float3[boneCount];
        _worldRotations = new quaternion[boneCount];
        _mirroredPositions = new float3[boneCount];
        _mirroredRotations = new quaternion[boneCount];
    }

    /// <summary>
    /// Pairs the skeleton's bones by name and finds its mirror plane from the rest pose. False when
    /// no left/right pair exists or the pairs do not agree on a side.
    /// </summary>
    public static bool TryCreate(Skeleton skeleton, out PoseMirror mirror, out string error)
    {
        mirror = null;
        if (skeleton == null || !skeleton.IsSet)
        {
            error = "No skeleton to mirror.";
            return false;
        }

        var data = skeleton.GetSkeletonData();
        var boneCount = data.BoneCount;

        var parents = new int[boneCount];
        var restWorldPositions = new float3[boneCount];
        var restWorldRotations = new quaternion[boneCount];
        for (var i = 0; i < boneCount; i++)
        {
            var parent = data.ParentIndices[i];
            parents[i] = parent;
            if (parent < 0)
            {
                restWorldPositions[i] = data.RestLocalPositions[i];
                restWorldRotations[i] = data.RestLocalRotations[i];
                continue;
            }

            restWorldRotations[i] = math.mul(restWorldRotations[parent], data.RestLocalRotations[i]);
            restWorldPositions[i] = restWorldPositions[parent] +
                                    math.rotate(restWorldRotations[parent], data.RestLocalPositions[i]);
        }

        var counterparts = new int[boneCount];
        var leftToRight = float3.zero;
        var pairCount = 0;
        for (var i = 0; i < boneCount; i++)
        {
            var name = skeleton.GetBone(i).Name;
            var counterpart = skeleton.IndexOfName(CounterpartName(name));
            counterparts[i] = counterpart >= 0 ? counterpart : i;

            if (counterpart >= 0 && counterpart != i && IsLeft(name))
            {
                leftToRight += restWorldPositions[i] - restWorldPositions[counterpart];
                pairCount++;
            }
        }

        if (pairCount == 0)
        {
            error = $"\"{skeleton.Name}\" has no left/right bone pair to mirror by: no bone name " +
                    "carries Left/Right or a .L/_L suffix with a matching counterpart.";
            return false;
        }

        var length = math.length(leftToRight);
        if (length < 1e-5f)
        {
            error = $"\"{skeleton.Name}\"'s {pairCount} left/right bone pairs cancel out at rest, so " +
                    "its mirror plane is undefined. Is the rest pose symmetric?";
            return false;
        }

        mirror = new PoseMirror(parents, counterparts, restWorldRotations, leftToRight / length);
        error = null;
        return true;
    }

    /// <summary>
    /// Writes the mirror image of <paramref name="source"/>'s positions and rotations into
    /// <paramref name="destination"/>. The two may be the same buffer; other channels are untouched.
    /// </summary>
    public void Mirror(PoseBuffer source, PoseBuffer destination)
    {
        var boneCount = _parents.Length;
        var sourcePositions = source.Positions;
        var sourceRotations = source.Rotations;
        if (sourcePositions.Length != boneCount || destination.Positions.Length != boneCount)
        {
            throw new ArgumentException(
                $"Pose has {sourcePositions.Length} bones but the mirror was built for {boneCount}.");
        }

        for (var i = 0; i < boneCount; i++)
        {
            var parent = _parents[i];
            if (parent < 0)
            {
                _worldPositions[i] = sourcePositions[i];
                _worldRotations[i] = sourceRotations[i];
                continue;
            }

            _worldRotations[i] = math.mul(_worldRotations[parent], sourceRotations[i]);
            _worldPositions[i] = _worldPositions[parent] + math.rotate(_worldRotations[parent], sourcePositions[i]);
        }

        for (var i = 0; i < boneCount; i++)
        {
            var counterpart = _counterparts[i];
            var deltaFromRest = math.mul(_worldRotations[counterpart], _restWorldRotationsInverse[counterpart]);
            _mirroredRotations[i] = math.mul(MirrorRotation(deltaFromRest), _restWorldRotations[i]);
            _mirroredPositions[i] = MirrorPoint(_worldPositions[counterpart]);
        }

        var positions = destination.Positions;
        var rotations = destination.Rotations;
        for (var i = 0; i < boneCount; i++)
        {
            var parent = _parents[i];
            if (parent < 0)
            {
                positions[i] = _mirroredPositions[i];
                rotations[i] = _mirroredRotations[i];
                continue;
            }

            var parentInverse = math.inverse(_mirroredRotations[parent]);
            rotations[i] = math.normalize(math.mul(parentInverse, _mirroredRotations[i]));
            positions[i] = math.rotate(parentInverse, _mirroredPositions[i] - _mirroredPositions[parent]);
        }
    }

    private float3 MirrorPoint(float3 point) => point - 2f * math.dot(point, _normal) * _normal;

    /// <summary>
    /// Conjugating a rotation by a reflection keeps its angle and reflects its axis, then flips the
    /// axis because a reflection reverses handedness.
    /// </summary>
    private quaternion MirrorRotation(quaternion rotation)
    {
        var axis = MirrorPoint(rotation.value.xyz);
        return new quaternion(new float4(-axis, rotation.value.w));
    }

    /// <summary>
    /// The name of the bone on the other side: Left/Right tokens anywhere in the name are swapped,
    /// and so is a trailing .L/.R or _L/_R. A bone is paired with its counterpart only when a bone
    /// of that name exists; otherwise it is its own counterpart.
    /// </summary>
    public static string CounterpartName(string name)
    {
        var swapped = SideToken.Replace(name, match => match.Value switch
        {
            "Left" => "Right",
            "Right" => "Left",
            "left" => "right",
            _ => "left"
        });

        return SideSuffix.Replace(swapped, match => match.Value[0] + (match.Value[1] == 'L' ? "R" : "L"));
    }

    private static bool IsLeft(string name)
    {
        var token = SideToken.Match(name);
        if (token.Success) return token.Value is "Left" or "left";

        var suffix = SideSuffix.Match(name);
        return suffix.Success && suffix.Value[1] == 'L';
    }
}
}
