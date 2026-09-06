using System;
using System.Collections.Generic;
using GameplayTags;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Tag channels as boolean keyframes, and the query that turns them into segments.
/// </summary>
/// <remarks>
/// A channel stores the frames its tag <em>flips</em> on, not the intervals it covers. That makes
/// every keyframe a handle with its own identity, which is what the clip editor's Blender-style
/// keymap grabs and moves, and it makes the data unfalsifiable: any sorted list of distinct frames
/// is a valid boolean signal, so there is no overlapping-or-inverted state to guard against the way
/// stored intervals would need.
/// <para>
/// Frames are slice-local, the same choice <see cref="GaitPhase.Footfall"/> documents, so the keys
/// stay correct when a clip's start or end frame moves.
/// </para>
/// </remarks>
public static class AnimationTagging
{
    /// <summary>One tag and the frames it switches on and off at.</summary>
    [Serializable]
    public sealed class TagChannel
    {
        public GameplayTagSO tag;

        [Tooltip("Frames the tag flips at, in this clip's sliced frame numbering. Sorted and " +
                 "distinct. The channel starts off, so an even-indexed key switches it on and an " +
                 "odd-indexed one switches it off.")]
        public List<int> toggles = new();
    }

    /// <summary>A derived interval a channel's tag is on over. Half-open, like a clip's own slice.</summary>
    public readonly struct TagSpan
    {
        public readonly int StartFrame;
        public readonly int EndFrame;

        public TagSpan(int startFrame, int endFrame)
        {
            StartFrame = startFrame;
            EndFrame = endFrame;
        }

        public int FrameCount => EndFrame - StartFrame;
    }

    /// <summary>
    /// Sorts the keys, cancels ones sharing a frame, and drops those the clip no longer holds.
    /// </summary>
    /// <remarks>
    /// Two keys on one frame describe a flip and an immediate flip back, which is the same signal
    /// as neither — and it is exactly what a drag that lands one key on another produces, so
    /// cancelling is the whole collision rule. An odd run leaves one key behind.
    /// <para>
    /// Dropping keys past the end can leave an odd number, i.e. the tag on to the end of the slice.
    /// That is the correct reading of what survived: it was on at the boundary.
    /// </para>
    /// </remarks>
    public static void Normalise(List<int> toggles, int frameCount)
    {
        if (toggles == null || toggles.Count == 0) return;

        toggles.Sort();

        var kept = 0;
        for (var i = 0; i < toggles.Count;)
        {
            var frame = toggles[i];

            var run = 1;
            while (i + run < toggles.Count && toggles[i + run] == frame) run++;

            if (run % 2 == 1 && frame >= 0 && frame <= frameCount) toggles[kept++] = frame;

            i += run;
        }

        toggles.RemoveRange(kept, toggles.Count - kept);
    }

    /// <summary>Whether the channel's tag is on at <paramref name="frame"/>.</summary>
    /// <remarks>
    /// A key switches the channel at its own frame, so the count is taken at or below it. Expects
    /// a normalised list.
    /// </remarks>
    public static bool IsOn(IReadOnlyList<int> toggles, int frame)
    {
        if (toggles == null) return false;

        var count = 0;
        for (var i = 0; i < toggles.Count && toggles[i] <= frame; i++) count++;

        return count % 2 == 1;
    }

    /// <summary>
    /// The intervals a channel is on over, pairing consecutive keys. A trailing unpaired key runs
    /// to <paramref name="frameCount"/>, so "on until the end" needs no closing key.
    /// </summary>
    public static void Spans(IReadOnlyList<int> toggles, int frameCount, List<TagSpan> results)
    {
        results.Clear();
        if (toggles == null) return;

        for (var i = 0; i < toggles.Count; i += 2)
        {
            var end = i + 1 < toggles.Count ? toggles[i + 1] : frameCount;
            if (end > toggles[i]) results.Add(new TagSpan(toggles[i], end));
        }
    }

    /// <summary>
    /// The runs of frames whose active tags satisfy <paramref name="query"/>, merged where they
    /// touch.
    /// </summary>
    /// <remarks>
    /// Sweeps the keys rather than the frames: the active tag set only changes where a channel
    /// flips, so the query is evaluated once per interval between keys instead of once per frame.
    /// </remarks>
    public static void FindSegments(AnnotatedAnimationClip clip, IReadOnlyList<TagChannel> channels,
        GameplayTagQuery query, int frameCount, List<AnimationClipSegment> results)
    {
        if (results == null) throw new ArgumentNullException(nameof(results));
        if (frameCount <= 0 || query == null) return;

        var boundaries = Boundaries(channels, frameCount);
        var active = new GameplayTagSet();

        var runStart = -1;
        for (var i = 0; i < boundaries.Count - 1; i++)
        {
            CollectTagsAt(channels, boundaries[i], active);

            if (query.Matches(active))
            {
                if (runStart < 0) runStart = boundaries[i];
                continue;
            }

            if (runStart >= 0) results.Add(new AnimationClipSegment(clip, runStart, boundaries[i]));
            runStart = -1;
        }

        if (runStart >= 0) results.Add(new AnimationClipSegment(clip, runStart, frameCount));
    }

    /// <summary>The tags on at <paramref name="frame"/>, replacing whatever <paramref name="into"/> held.</summary>
    public static void CollectTagsAt(IReadOnlyList<TagChannel> channels, int frame,
        GameplayTagSet into)
    {
        into.Clear();
        if (channels == null) return;

        foreach (var channel in channels)
        {
            if (channel?.tag != null && IsOn(channel.toggles, frame)) into.AddTag(channel.tag);
        }
    }

    /// <summary>Every frame the active tag set can change at, plus both ends, sorted and distinct.</summary>
    private static List<int> Boundaries(IReadOnlyList<TagChannel> channels, int frameCount)
    {
        var boundaries = new List<int> { 0, frameCount };

        if (channels != null)
        {
            foreach (var channel in channels)
            {
                if (channel?.toggles == null) continue;

                foreach (var toggle in channel.toggles)
                {
                    if (toggle > 0 && toggle < frameCount) boundaries.Add(toggle);
                }
            }
        }

        boundaries.Sort();

        var kept = 1;
        for (var i = 1; i < boundaries.Count; i++)
        {
            if (boundaries[i] != boundaries[kept - 1]) boundaries[kept++] = boundaries[i];
        }

        boundaries.RemoveRange(kept, boundaries.Count - kept);
        return boundaries;
    }
}
}
