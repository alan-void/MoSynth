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
    /// Bones whose contact the database flags, one channel each, in slot order. Empty means no
    /// contact channels; nothing falls back to a name heuristic. An unset entry is null.
    /// </summary>
    IReadOnlyList<string> ContactBoneNames { get; }

    /// <summary>
    /// Also bake each clip mirrored left-to-right, as a clip of its own right after the original.
    /// </summary>
    bool MirrorClips { get; }

    /// <summary>Directory the serialized database lives in. Created if missing (editor only).</summary>
    string GetAssetPath();
}
}
