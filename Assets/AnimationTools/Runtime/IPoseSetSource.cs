using System.Collections.Generic;
using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Everything the pose-database pipeline needs from the asset it is extracting, so any config can
/// supply a pose database without being a full motion matching asset.
/// </summary>
public interface IPoseSetSource
{
    /// <summary>Asset name. Doubles as the database folder and file base name.</summary>
    string name { get; }

    /// <summary>
    /// The pose skeleton, identical to every clip's skeleton: bone 0 is the rig's root and carries
    /// the clip's world position and rotation.
    /// </summary>
    Skeleton Skeleton { get; }

    /// <summary>Clips the database is extracted from, in the order they are stored.</summary>
    List<AnnotatedAnimationClip> AnimationClips { get; }

    /// <summary>Foot speed below which a toe counts as planted.</summary>
    float ContactVelocityThreshold { get; }

    /// <summary>
    /// Bone whose velocity drives left foot-contact detection during extraction; null/empty
    /// picks by name heuristic.
    /// </summary>
    string LeftContactBoneName { get; }

    /// <summary>
    /// Bone whose velocity drives right foot-contact detection during extraction; null/empty
    /// picks by name heuristic.
    /// </summary>
    string RightContactBoneName { get; }

    /// <summary>Directory the serialized database lives in. Created if missing (editor only).</summary>
    string GetAssetPath();
}
}
