using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// One dataset's batch-retarget run: which setup blend replays over which BVH folder, and
/// where the resulting FBX lands. Stored as an asset so a run is repeatable and reviewable
/// rather than retyped into a window each time.
/// </summary>
/// <remarks>
/// Everything about <em>how</em> a source skeleton retargets lives in the blend itself; see
/// Tools/Retargeting/README.md.
/// </remarks>
[CreateAssetMenu(fileName = "RetargetBatch", menuName = "MoSynth/Retarget Batch Settings")]
public class RetargetBatchSettings : ScriptableObject
{
    [Tooltip("The setup .blend for this source skeleton, as a project-relative path.")]
    public string setupBlendPath = "Assets/LFS/Retargeting/retargeting.blend";

    [Tooltip("Folder holding the source BVH files, as a project-relative path.")]
    public string bvhFolderPath = "Assets/LFS/Animation/lafan1/bvh";

    [Tooltip("Glob matched against file names in the BVH folder. One setup blend covers one " +
             "source skeleton, so this selects the clips that belong to it.")]
    public string filePattern = "*.bvh";

    [Tooltip("Destination FBX, as a project-relative path. Every matched clip becomes a take " +
             "in this one file.")]
    public string outputFbxPath = "Assets/LFS/Animation/lafan1/retargeted/lafan1.fbx";

    [Min(0f)]
    [Tooltip("FBX keyframe reduction. 0 keeps every key. On a 7840-frame LAFAN clip, 1 costs " +
             "about 0.2 degrees and saves roughly two thirds of the file size.")]
    public float simplifyFactor;

    [Min(0)]
    [Tooltip("Process at most this many clips; 0 means all of them. Useful for a smoke run " +
             "before committing to a multi-hour batch.")]
    public int clipLimit;
}
}
