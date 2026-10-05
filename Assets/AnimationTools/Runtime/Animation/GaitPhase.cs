using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Turns a clip's footfalls into a gait phase — the angle a phase-functioned network is organised
/// around, saying where in the walk cycle a frame sits.
/// </summary>
/// <remarks>
/// This is the project's only definition of phase: evaluated here, written into the pose database,
/// and read back from there by the training set. Half a cycle per footfall, a whole cycle when the
/// same foot falls twice running (which is what a missed contact looks like), linear between
/// anchors, and the cycle zeroed on a right footfall.
/// <para>
/// A stretch with no footfalls is answered frame by frame by ground speed, since standing and a
/// missed contact look alike in the anchors. A standing frame sweeps at a fixed rate so a model can
/// learn phase does not matter there; a moving one is held at rate zero, marking it unusable,
/// because extrapolating a walking rate over a stand invents gait. See
/// openwiki/animation-tools/neural-synthesis.md.
/// </para>
/// </remarks>
public static class GaitPhase
{
    /// <summary>Which foot a footfall belongs to. Ordered so Left is 0, matching the contact channels.</summary>
    public enum Foot
    {
        Left,
        Right
    }

    /// <summary>
    /// The frame a foot was planted on, numbered against the whole baked clip.
    /// </summary>
    /// <remarks>
    /// Clip-local, not slice-local, so trimming a clip cannot change which moment of the animation
    /// an anchor names.
    /// </remarks>
    [Serializable]
    public struct Footfall
    {
        public int frame;
        public Foot foot;

        public Footfall(int frame, Foot foot)
        {
            this.frame = frame;
            this.foot = foot;
        }
    }

    /// <summary>A full cycle.</summary>
    public const float Tau = 2f * math.PI;

    /// <summary>
    /// Fills <paramref name="phase"/> and <paramref name="phaseRate"/> from a clip's footfalls,
    /// with no measure of speed, so every stretch outside the anchors is held at a rate of zero.
    /// </summary>
    public static void Evaluate(IReadOnlyList<Footfall> footfalls, float frameTime,
        float[] phase, float[] phaseRate)
        => Evaluate(footfalls, frameTime, null, 0f, 0f, phase, phaseRate);

    /// <summary>
    /// Fills <paramref name="phase"/> and <paramref name="phaseRate"/> from a clip's footfalls and
    /// how fast the character was travelling.
    /// </summary>
    /// <param name="footfalls">Anchors in ascending frame order.</param>
    /// <param name="frameTime">Seconds per frame, so the rate comes out per second.</param>
    /// <param name="speed">
    /// The character frame's ground speed per frame, in m/s, indexed as <paramref name="phase"/> is.
    /// Null switches the standing rule off entirely.
    /// </param>
    /// <param name="standingSpeed">Speed below which a frame counts as standing.</param>
    /// <param name="standingPeriod">
    /// Seconds per cycle a standing stretch sweeps at. Zero switches the standing rule off.
    /// </param>
    /// <param name="phase">Filled with radians in <c>[0, Tau)</c>. Length sets the frame count.</param>
    /// <param name="phaseRate">
    /// Filled with radians per second. <b>Zero marks a frame with no measurable cycle</b> — one
    /// with no footfalls around it that the character was moving through — which is the
    /// sentinel training uses to drop a frame. A standing frame reports the standing rate and is
    /// kept.
    /// </param>
    public static void Evaluate(IReadOnlyList<Footfall> footfalls, float frameTime,
        IReadOnlyList<float> speed, float standingSpeed, float standingPeriod,
        float[] phase, float[] phaseRate)
    {
        if (phase == null) throw new ArgumentNullException(nameof(phase));
        if (phaseRate == null) throw new ArgumentNullException(nameof(phaseRate));
        if (phase.Length != phaseRate.Length)
        {
            throw new ArgumentException("phase and phaseRate must be the same length.");
        }

        Array.Clear(phase, 0, phase.Length);
        Array.Clear(phaseRate, 0, phaseRate.Length);

        if (frameTime <= 0f) return;

        // Radians per frame while standing. Zero is the switch that leaves every anchorless stretch
        // held, which is what a caller with no speed measurement gets.
        var standingSlope = speed != null && standingPeriod > 0f
            ? Tau * frameTime / standingPeriod
            : 0f;

        var anchors = UsableAnchors(footfalls, phase.Length);
        if (anchors.Count == 0)
        {
            // A clip the character stood through still teaches a stationary pose, so it sweeps from
            // zero -- there is no anchor to line up with. Frames it moved through are ones whose
            // contacts were missed, and stay unusable.
            SweepForward(phase, phaseRate, 0, phase.Length, 0f,
                speed, standingSpeed, standingSlope, frameTime);
            return;
        }

        var targets = UnwrappedTargets(anchors, speed, standingSpeed, standingSlope);

        for (var i = 0; i < anchors.Count - 1; i++)
        {
            var from = anchors[i].frame;
            var to = anchors[i + 1].frame;
            var span = to - from;
            if (span <= 0) continue;

            var slope = (targets[i + 1] - targets[i]) / span;
            var rate = slope / frameTime;

            for (var frame = from; frame < to; frame++)
            {
                phase[frame] = Wrap(targets[i] + slope * (frame - from));
                phaseRate[frame] = rate;
            }
        }

        // The lead-in and lead-out have no second anchor, so each frame answers for itself: standing
        // frames sweep at the standing rate from the footfall they meet, moving ones hold at rate
        // zero. Per frame, because a clip that stands then walks off accelerates before its first
        // heel strike, and judging the stretch whole would discard the stand.
        var firstFrame = anchors[0].frame;
        var lastFrame = anchors[anchors.Count - 1].frame;
        var lastTarget = targets[targets.Count - 1];

        SweepBackward(phase, phaseRate, 0, firstFrame, targets[0],
            speed, standingSpeed, standingSlope, frameTime);
        SweepForward(phase, phaseRate, lastFrame, phase.Length, lastTarget,
            speed, standingSpeed, standingSlope, frameTime);
    }

    /// <summary>
    /// Whether one frame was slow enough to count as standing. False when nothing measured speed or
    /// the standing rule is switched off, so those callers keep the held-and-unusable answer.
    /// </summary>
    private static bool IsStandingFrame(IReadOnlyList<float> speed, int frame,
        float standingSpeed, float standingSlope) =>
        speed != null && standingSlope > 0f && frame < speed.Count && speed[frame] < standingSpeed;

    /// <summary>How many frames of <c>[from, to)</c> the character stood through.</summary>
    private static int StandingFrames(IReadOnlyList<float> speed, int from, int to,
        float standingSpeed, float standingSlope)
    {
        var standing = 0;
        for (var frame = from; frame < to; frame++)
        {
            if (IsStandingFrame(speed, frame, standingSpeed, standingSlope)) standing++;
        }

        return standing;
    }

    /// <summary>
    /// Fills <c>[from, to)</c> running forward from an unwrapped target: a standing frame advances
    /// the phase at the standing rate, a moving one holds it and reports a rate of zero.
    /// </summary>
    private static void SweepForward(float[] phase, float[] phaseRate, int from, int to,
        float startTarget, IReadOnlyList<float> speed, float standingSpeed, float standingSlope,
        float frameTime)
    {
        var rate = standingSlope / frameTime;
        var target = startTarget;

        for (var frame = from; frame < to; frame++)
        {
            phase[frame] = Wrap(target);
            if (!IsStandingFrame(speed, frame, standingSpeed, standingSlope)) continue;

            phaseRate[frame] = rate;
            target += standingSlope;
        }
    }

    /// <summary>
    /// Fills <c>[from, to)</c> running backward from the unwrapped target the frame at
    /// <paramref name="to"/> lands on, so the sweep meets the anchor it runs up to.
    /// </summary>
    private static void SweepBackward(float[] phase, float[] phaseRate, int from, int to,
        float endTarget, IReadOnlyList<float> speed, float standingSpeed, float standingSlope,
        float frameTime)
    {
        var rate = standingSlope / frameTime;
        var target = endTarget;

        for (var frame = to - 1; frame >= from; frame--)
        {
            if (IsStandingFrame(speed, frame, standingSpeed, standingSlope))
            {
                target -= standingSlope;
                phaseRate[frame] = rate;
            }

            phase[frame] = Wrap(target);
        }
    }

    /// <summary>
    /// Anchors inside the clip, in ascending frame order, with any repeated frame dropped.
    /// </summary>
    /// <remarks>
    /// Two footfalls on one frame would make a zero-length segment, so the first is kept. Anchors
    /// outside the clip are skipped here, which is why nothing needs to delete them.
    /// </remarks>
    private static List<Footfall> UsableAnchors(IReadOnlyList<Footfall> footfalls, int frameCount)
    {
        var anchors = new List<Footfall>();
        if (footfalls == null) return anchors;

        foreach (var footfall in footfalls)
        {
            if (footfall.frame < 0 || footfall.frame >= frameCount) continue;
            if (anchors.Count > 0 && footfall.frame <= anchors[anchors.Count - 1].frame) continue;
            anchors.Add(footfall);
        }

        return anchors;
    }

    /// <summary>
    /// The unwrapped phase each anchor lands on: half a cycle when the feet alternate, a whole one
    /// when the same foot falls twice, which is how a missed contact shows up.
    /// </summary>
    private static List<float> UnwrappedTargets(IReadOnlyList<Footfall> anchors,
        IReadOnlyList<float> speed, float standingSpeed, float standingSlope)
    {
        var targets = new List<float>(anchors.Count) { 0f };
        for (var i = 1; i < anchors.Count; i++)
        {
            var step = anchors[i].foot != anchors[i - 1].foot ? math.PI : Tau;

            // A gap the character stood through is not one slow stride: the step grows by whole
            // cycles, as near the standing rate as still landing on the anchor with the right foot
            // allows. Standing is counted per frame, since a mid-clip stand is bracketed by
            // deceleration and acceleration that would hide it if the whole gap had to be slow.
            var standing = StandingFrames(speed, anchors[i - 1].frame, anchors[i].frame,
                standingSpeed, standingSlope);
            if (standing > 0)
            {
                step += math.max(0f, math.round((standing * standingSlope - step) / Tau)) * Tau;
            }

            targets.Add(targets[i - 1] + step);
        }

        // The cycle is conventionally zeroed on a right footfall, so a clip that opens on a left one
        // is shifted by half a cycle. Without this, "phase 0" would mean a different pose depending
        // on which foot a clip happened to start with.
        if (anchors[0].foot == Foot.Left)
        {
            for (var i = 0; i < targets.Count; i++) targets[i] += math.PI;
        }

        return targets;
    }

    private static float Wrap(float unwrapped)
    {
        var wrapped = unwrapped - math.floor(unwrapped / Tau) * Tau;

        // floor() of a value a hair below a multiple of Tau can still land on it, which would return
        // exactly Tau and break the half-open range every consumer assumes.
        return wrapped >= Tau ? 0f : wrapped;
    }

    /// <summary>
    /// Median-filters each foot's contact flags in place, so a frame counts as planted when most of
    /// the window around it was.
    /// </summary>
    /// <remarks>
    /// Removes single-frame chatter, at the cost of moving a footfall's detected frame by up to the
    /// radius and erasing any stance shorter than half the window. Edges are clamped.
    /// </remarks>
    /// <param name="contacts">Flags interleaved per frame: <c>frame * 2</c> left, <c>+ 1</c> right.</param>
    public static void SmoothContacts(bool[] contacts, int frameCount, int radius) =>
        SmoothInterleaved(contacts, frameCount, radius, stride: 2);

    /// <summary>
    /// As <see cref="SmoothContacts"/>, for one bone's flags: index <c>frame</c> is that frame.
    /// </summary>
    public static void SmoothContactTrack(bool[] contacts, int frameCount, int radius) =>
        SmoothInterleaved(contacts, frameCount, radius, stride: 1);

    private static void SmoothInterleaved(bool[] contacts, int frameCount, int radius, int stride)
    {
        if (contacts == null || radius <= 0 || frameCount <= 0) return;

        var source = (bool[])contacts.Clone();
        var window = radius * 2 + 1;

        for (var track = 0; track < stride; track++)
        {
            for (var frame = 0; frame < frameCount; frame++)
            {
                var planted = 0;
                for (var offset = -radius; offset <= radius; offset++)
                {
                    var sample = math.clamp(frame + offset, 0, frameCount - 1);
                    if (source[sample * stride + track]) planted++;
                }

                contacts[frame * stride + track] = planted * 2 > window;
            }
        }
    }

    /// <summary>
    /// The frames each foot went down on, merged into one ascending list.
    /// </summary>
    /// <remarks>
    /// A foot already planted on frame 0 is not a footfall: the clip started mid-stance and nothing
    /// says when that foot went down, so anchoring the cycle there would be a guess.
    /// </remarks>
    public static List<Footfall> FootfallsFromContacts(bool[] contacts, int frameCount)
    {
        var footfalls = new List<Footfall>();
        if (contacts == null) return footfalls;

        for (var frame = 1; frame < frameCount; frame++)
        {
            if (contacts[frame * 2] && !contacts[(frame - 1) * 2])
            {
                footfalls.Add(new Footfall(frame, Foot.Left));
            }

            if (contacts[frame * 2 + 1] && !contacts[(frame - 1) * 2 + 1])
            {
                footfalls.Add(new Footfall(frame, Foot.Right));
            }
        }

        footfalls.Sort((a, b) => a.frame.CompareTo(b.frame));
        return footfalls;
    }

    /// <summary>
    /// How many anchors are the same foot twice running — each one a footfall the contact detection
    /// missed, and a place the phase jumps a whole cycle instead of half.
    /// </summary>
    public static int CountRepeatedFeet(IReadOnlyList<Footfall> footfalls)
    {
        if (footfalls == null) return 0;

        var repeats = 0;
        for (var i = 1; i < footfalls.Count; i++)
        {
            if (footfalls[i].foot == footfalls[i - 1].foot) repeats++;
        }

        return repeats;
    }
}
}
