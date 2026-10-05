using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// What a clip's gait is measured from: whether each foot is planted, and how fast the character is
/// travelling, frame by frame.
/// </summary>
/// <remarks>
/// Both readings difference character-space positions across consecutive frames, and both are used
/// by the clip editor and by the pose-database bake, so what an author corrects a clip against is
/// what the database records.
/// <para>
/// Differencing positions is the measurement to trust over composing velocity channels; see
/// openwiki/animation-tools/animation-sources.md.
/// </para>
/// </remarks>
public static class GaitMeasure
{
    /// <summary>Half-width of the median filter over raw contact flags, in frames.</summary>
    public const int DefaultSmoothingRadius = 6;

    /// <summary>Foot speed below which a toe counts as planted, in m/s.</summary>
    public const float DefaultContactVelocityThreshold = 0.15f;

    /// <summary>
    /// Whether each foot is planted on each of <paramref name="frameCount"/> frames from
    /// <paramref name="firstFrame"/>, smoothed. Index <c>i * 2</c> is left, <c>+ 1</c> right.
    /// </summary>
    public static bool[] Contacts(SkeletonAnimation clip, Skeleton skeleton, int leftBone, int rightBone,
        int firstFrame, int frameCount, float velocityThreshold, int smoothingRadius)
    {
        var contacts = new bool[math.max(0, frameCount) * 2];
        var left = Contacts(clip, skeleton, leftBone, firstFrame, frameCount, velocityThreshold, smoothingRadius);
        var right = Contacts(clip, skeleton, rightBone, firstFrame, frameCount, velocityThreshold, smoothingRadius);
        for (var i = 0; i < left.Length; i++)
        {
            contacts[i * 2] = left[i];
            contacts[i * 2 + 1] = right[i];
        }

        return contacts;
    }

    /// <summary>
    /// Whether <paramref name="bone"/> is planted on each of <paramref name="frameCount"/> frames
    /// from <paramref name="firstFrame"/>, smoothed. Index <c>i</c> is frame <c>firstFrame + i</c>.
    /// </summary>
    public static bool[] Contacts(SkeletonAnimation clip, Skeleton skeleton, int bone,
        int firstFrame, int frameCount, float velocityThreshold, int smoothingRadius)
    {
        var contacts = new bool[math.max(0, frameCount)];
        if (frameCount < 2 || clip.FrameTime <= 0f) return contacts;

        var positions = new float3[frameCount];
        var skeletonData = skeleton.GetSkeletonData();
        for (var i = 0; i < frameCount; i++)
        {
            positions[i] = skeletonData.CharacterSpacePosition(clip.GetFrame(firstFrame + i), bone);
        }

        for (var i = 0; i < frameCount; i++)
        {
            // The last frame has no successor to difference against, so it inherits the one before
            // it rather than being reported as a sudden plant.
            contacts[i] = Speed(positions, math.min(i, frameCount - 2), clip.FrameTime) < velocityThreshold;
        }

        GaitPhase.SmoothContactTrack(contacts, frameCount, smoothingRadius);
        return contacts;
    }

    /// <summary>
    /// The character frame's ground speed in m/s, for each of <paramref name="frameCount"/> frames
    /// from <paramref name="firstFrame"/>. This is what separates a stand from a missed contact.
    /// </summary>
    public static float[] GroundSpeed(SkeletonAnimation clip, Skeleton skeleton, int firstFrame, int frameCount)
    {
        var speed = new float[math.max(0, frameCount)];
        if (frameCount < 2 || clip.FrameTime <= 0f) return speed;

        var positions = new float3[frameCount];
        var skeletonData = skeleton.GetSkeletonData();
        var frameDef = SimulationFrameDef.Default(skeleton);
        for (var i = 0; i < frameCount; i++)
        {
            SimulationFrame.Compute(clip.GetFrame(firstFrame + i), skeletonData, frameDef,
                out positions[i], out _);
        }

        for (var i = 0; i < frameCount; i++)
        {
            speed[i] = Speed(positions, math.min(i, frameCount - 2), clip.FrameTime);
        }

        return speed;
    }

    private static float Speed(float3[] positions, int from, float frameTime) =>
        math.length(positions[from + 1] - positions[from]) / frameTime;
}
}
