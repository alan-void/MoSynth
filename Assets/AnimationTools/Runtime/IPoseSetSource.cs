using System.Collections.Generic;
using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Everything the pose-database pipeline needs from the asset it is extracting.
///
/// <see cref="PoseSet"/>, <see cref="PoseExtractor"/> and <see cref="PoseSerializer"/> used to take
/// a <see cref="MotionMatchingData"/> directly, which meant anything wanting a pose database also
/// had to be a full Motion Matching asset -- trajectory features, pose features, skeleton map and
/// all. Only a handful of those members actually reach the pose path, and this interface is
/// exactly that handful, so a consumer such as MotionField's config can supply a database without
/// dragging in the feature-set machinery it never uses.
///
/// This lives in the AnimationTools runtime assembly since it is generic pose-database
/// infrastructure, not specific to any one motion-synthesis technique.
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
