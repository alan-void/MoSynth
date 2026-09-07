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
/// Differencing rather than composing the per-bone velocity channels is not only a convenience — a
/// clip's baked poses carry positions and rotations only — it is also the measurement to trust. The
/// composed form the bake once used disagreed badly, reporting a toe planted 0.22/0.17 of the time
/// on <c>walk1_subject5</c> where differencing reports 0.40/0.36.
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
        if (frameCount < 2 || clip.FrameTime <= 0f) return contacts;

        var left = new float3[frameCount];
        var right = new float3[frameCount];
        var skeletonData = skeleton.GetSkeletonData();
        for (var i = 0; i < frameCount; i++)
        {
            var pose = clip.GetFrame(firstFrame + i);
            left[i] = skeletonData.CharacterSpacePosition(pose, leftBone);
            right[i] = skeletonData.CharacterSpacePosition(pose, rightBone);
        }

        for (var i = 0; i < frameCount; i++)
        {
            // The last frame has no successor to difference against, so it inherits the one before
            // it rather than being reported as a sudden plant.
            var from = math.min(i, frameCount - 2);
            contacts[i * 2] = Speed(left, from, clip.FrameTime) < velocityThreshold;
            contacts[i * 2 + 1] = Speed(right, from, clip.FrameTime) < velocityThreshold;
        }

        GaitPhase.SmoothContacts(contacts, frameCount, smoothingRadius);
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
