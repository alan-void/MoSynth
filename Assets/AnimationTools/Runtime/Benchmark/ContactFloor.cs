using System;
using System.Collections.Generic;

namespace AnimationTools
{
/// <summary>
/// Where each contact bone sits when planted, by bone name, taken from a pose database rather than
/// from the run being judged. Assumes flat ground at y = 0.
/// </summary>
public sealed class ContactFloor
{
    /// <summary>Quantile of a bone's flagged database heights taken as the height it rests at.</summary>
    public const float RestQuantile = 0.05f;

    private readonly string[] _boneNames;
    private readonly float[] _restHeights;
    private readonly float[] _meanContactHeights;

    public ContactFloor(IReadOnlyList<string> boneNames, IReadOnlyList<float> restHeights,
        IReadOnlyList<float> meanContactHeights)
    {
        if (boneNames == null) throw new ArgumentNullException(nameof(boneNames));
        if (restHeights == null) throw new ArgumentNullException(nameof(restHeights));
        if (meanContactHeights == null) throw new ArgumentNullException(nameof(meanContactHeights));
        if (restHeights.Count != boneNames.Count || meanContactHeights.Count != boneNames.Count)
            throw new ArgumentException("Every bone name needs one rest height and one mean contact height.");

        _boneNames = new string[boneNames.Count];
        _restHeights = new float[boneNames.Count];
        _meanContactHeights = new float[boneNames.Count];
        for (var i = 0; i < _boneNames.Length; i++)
        {
            _boneNames[i] = boneNames[i];
            _restHeights[i] = restHeights[i];
            _meanContactHeights[i] = meanContactHeights[i];
        }
    }

    /// <summary>Contact bone names, in the database's slot order.</summary>
    public IReadOnlyList<string> BoneNames => _boneNames;

    /// <summary>
    /// Each bone's floor: the <see cref="RestQuantile"/> quantile of its height over the database frames
    /// flagged in contact, meters. NaN for a bone never flagged.
    /// </summary>
    public IReadOnlyList<float> RestHeights => _restHeights;

    /// <summary>Each bone's mean height over the database frames flagged in contact, meters. NaN for a bone never flagged.</summary>
    public IReadOnlyList<float> MeanContactHeights => _meanContactHeights;

    /// <summary>
    /// Both heights of <paramref name="boneName"/>; false when the bone is not a contact bone of the
    /// database or was never flagged in contact there.
    /// </summary>
    public bool TryGetHeights(string boneName, out float restHeight, out float meanContactHeight)
    {
        var index = Array.IndexOf(_boneNames, boneName);
        restHeight = index >= 0 ? _restHeights[index] : float.NaN;
        meanContactHeight = index >= 0 ? _meanContactHeights[index] : float.NaN;
        return index >= 0 && !float.IsNaN(restHeight);
    }

    /// <remarks>
    /// Stored poses are in clip space, where bone 0 carries the clip's world position, so a height there
    /// is height above the capture floor: the same world height a run records on ground at y = 0. The
    /// floor is a low quantile rather than the mean because flags also cover frames where the toe is
    /// still rolling on or off the ground.
    /// </remarks>
    public static ContactFloor FromDatabase(PoseSet poseSet)
    {
        if (poseSet == null) throw new ArgumentNullException(nameof(poseSet));

        var contacts = poseSet.ContactHandles;
        var skeletonData = poseSet.Skeleton.GetSkeletonData();
        var flaggedHeights = new List<float>[contacts.Count];
        for (var slot = 0; slot < flaggedHeights.Length; slot++) flaggedHeights[slot] = new List<float>();

        for (var poseIndex = 0; poseIndex < poseSet.NumberPoses; poseIndex++)
        {
            var pose = poseSet.GetPoseBuffer(poseIndex);
            for (var slot = 0; slot < contacts.Count; slot++)
            {
                if (!pose.GetBool(contacts[slot])) continue;

                flaggedHeights[slot].Add(skeletonData.CharacterSpacePosition(pose, contacts.GetBoneIndex(slot)).y);
            }
        }

        var restHeights = new float[contacts.Count];
        var meanContactHeights = new float[contacts.Count];
        for (var slot = 0; slot < contacts.Count; slot++)
        {
            restHeights[slot] = MetricsMath.Percentile(flaggedHeights[slot], RestQuantile);
            meanContactHeights[slot] = MetricsMath.Mean(flaggedHeights[slot]);
        }

        return new ContactFloor(poseSet.ContactBoneNames, restHeights, meanContactHeights);
    }
}
}
