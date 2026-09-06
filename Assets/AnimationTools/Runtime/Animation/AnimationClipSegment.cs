namespace AnimationTools
{
/// <summary>
/// A contiguous run of frames within one clip, in that clip's sliced frame numbering.
/// </summary>
/// <remarks>
/// Nothing about naming part of a clip is specific to tags, so this is the shared result type for
/// anything that answers "which parts of these clips". Three neighbouring words mean three
/// different things here and are not interchangeable: a <em>slice</em> is a clip's own
/// <c>[startFrame, endFrame)</c> window, a <em>span</em> is one tag channel's on-interval, and a
/// <em>segment</em> is a run of frames some question selected.
/// </remarks>
public readonly struct AnimationClipSegment
{
    public readonly AnnotatedAnimationClip Clip;

    /// <summary>Inclusive, in <see cref="Clip"/>'s sliced frame numbering.</summary>
    public readonly int StartFrame;

    /// <summary>Exclusive, in <see cref="Clip"/>'s sliced frame numbering.</summary>
    public readonly int EndFrame;

    public AnimationClipSegment(AnnotatedAnimationClip clip, int startFrame, int endFrame)
    {
        Clip = clip;
        StartFrame = startFrame;
        EndFrame = endFrame;
    }

    public int FrameCount => EndFrame - StartFrame;

    public override string ToString() =>
        $"{(Clip == null ? "(no clip)" : Clip.name)} [{StartFrame}, {EndFrame})";
}
}
