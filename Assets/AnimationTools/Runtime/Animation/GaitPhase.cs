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
/// The rule is the one <c>Python/gait_phase.py</c> uses, because a model trained on one definition
/// and run on another does not throw, it just moves badly: half a cycle per footfall, a whole cycle
/// when the same foot falls twice running (which is what a missed contact looks like), linear
/// between anchors, and the cycle zeroed on a right footfall.
/// <para>
/// One deliberate difference. Python extrapolates outward from the first and last anchor so a clip
/// has no flat ends; this holds the phase still there and reports a rate of zero. Extrapolating
/// invents gait: on the untrimmed <c>walk1_subject1</c> clip the first real footfall is at frame
/// 132, and the 4.4 s of standing before it were being given 2.87 complete cycles of phase that the
/// character never walked. A model trained on that learns to cycle its legs while stationary.
/// </para>
/// <para>
/// The rate is likewise the exact slope of the segment a frame sits in, rather than Python's
/// central difference of the unwrapped phase, which smears one frame of ramp across every anchor.
/// The two agree everywhere except at the anchors themselves.
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
    /// Clip-local, not slice-local, so that trimming a clip cannot change which moment of the
    /// animation an anchor names. Slice-local addressing is measured from <c>startFrame</c>, so
    /// moving the start silently slides every marker across the motion - the encoding least able to
    /// survive the edit it was once claimed to survive.
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
    /// Fills <paramref name="phase"/> and <paramref name="phaseRate"/> from a clip's footfalls.
    /// </summary>
    /// <param name="footfalls">Anchors in ascending frame order. Fewer than two means no cycle.</param>
    /// <param name="frameTime">Seconds per frame, so the rate comes out per second.</param>
    /// <param name="phase">Filled with radians in <c>[0, Tau)</c>. Length sets the frame count.</param>
    /// <param name="phaseRate">
    /// Filled with radians per second. <b>Zero marks a frame with no measurable cycle</b> — outside
    /// the anchors, or on a clip that has none — which is the sentinel training uses to drop a
    /// frame.
    /// </param>
    public static void Evaluate(IReadOnlyList<Footfall> footfalls, float frameTime,
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

        var anchors = UsableAnchors(footfalls, phase.Length);
        if (anchors.Count < 2 || frameTime <= 0f) return;

        var targets = UnwrappedTargets(anchors);

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

        // The outer frames hold the boundary anchors' phase at a rate of zero: the clip contains no
        // evidence of a cycle there, and inventing one is the failure this exists to avoid.
        var firstFrame = anchors[0].frame;
        var lastFrame = anchors[anchors.Count - 1].frame;

        for (var frame = 0; frame < firstFrame; frame++) phase[frame] = Wrap(targets[0]);
        for (var frame = lastFrame; frame < phase.Length; frame++)
        {
            phase[frame] = Wrap(targets[targets.Count - 1]);
        }
    }

    /// <summary>
    /// Anchors inside the clip, in ascending frame order, with any repeated frame dropped.
    /// </summary>
    /// <remarks>
    /// Two footfalls on one frame would make a zero-length segment, and the phase across it
    /// undefined. Keeping the first is arbitrary but total; the alternative is a divide by zero,
    /// which is what the Python side does today on such a clip.
    /// <para>
    /// This filter is also why nothing needs to delete an anchor for falling outside the clip: one
    /// that does is skipped here and costs nothing. An <c>OnValidate</c> that removed them instead
    /// destroyed a clip's gait data every time its range was touched.
    /// </para>
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
    private static List<float> UnwrappedTargets(IReadOnlyList<Footfall> anchors)
    {
        var targets = new List<float>(anchors.Count) { 0f };
        for (var i = 1; i < anchors.Count; i++)
        {
            var step = anchors[i].foot != anchors[i - 1].foot ? math.PI : Tau;
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
    /// Matches the filter <c>PoseExtractor.SmoothContacts</c> applies at bake time, edge clamping
    /// included. It removes single-frame chatter, at the cost of moving a footfall's detected frame
    /// by up to the radius and of erasing any stance shorter than half the window.
    /// </remarks>
    /// <param name="contacts">Flags interleaved per frame: <c>frame * 2</c> left, <c>+ 1</c> right.</param>
    public static void SmoothContacts(bool[] contacts, int frameCount, int radius)
    {
        if (contacts == null || radius <= 0 || frameCount <= 0) return;

        var source = (bool[])contacts.Clone();
        var window = radius * 2 + 1;

        for (var foot = 0; foot < 2; foot++)
        {
            for (var frame = 0; frame < frameCount; frame++)
            {
                var planted = 0;
                for (var offset = -radius; offset <= radius; offset++)
                {
                    var sample = math.clamp(frame + offset, 0, frameCount - 1);
                    if (source[sample * 2 + foot]) planted++;
                }

                contacts[frame * 2 + foot] = planted * 2 > window;
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
