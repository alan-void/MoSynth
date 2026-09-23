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
/// A channel stores the frames its tag <em>flips</em> on, not intervals: every key is an editable
/// handle, and any sorted list of distinct frames is a valid signal. Frames are numbered against the
/// whole clip, so trimming never changes what a key names. See openwiki/animation-tools/clip-tags.md.
/// </remarks>
public static class AnimationTagging
{
    /// <summary>One tag and the frames it switches on and off at.</summary>
    [Serializable]
    public sealed class TagChannel
    {
        public GameplayTagSO tag;

        [Tooltip("Frames the tag flips at, numbered against the whole clip. Sorted and distinct. " +
                 "The channel starts off, so an even-indexed key switches it on and an odd-indexed " +
                 "one switches it off.")]
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

    /// <summary>Sorts the keys and cancels ones sharing a frame.</summary>
    /// <remarks>
    /// Two keys on one frame are a flip and a flip back, the same signal as neither, so they cancel;
    /// an odd run leaves one. Keys outside the clip's current range are kept, since trimming must
    /// not destroy annotation. Negative frames are dropped: <see cref="IsOn"/> would count them at
    /// every frame, inverting the channel.
    /// </remarks>
    public static void Normalise(List<int> toggles)
    {
        if (toggles == null || toggles.Count == 0) return;

        toggles.Sort();

        var kept = 0;
        for (var i = 0; i < toggles.Count;)
        {
            var frame = toggles[i];

            var run = 1;
            while (i + run < toggles.Count && toggles[i + run] == frame) run++;

            if (run % 2 == 1 && frame >= 0) toggles[kept++] = frame;

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
    /// to <paramref name="clipFrameCount"/>, so "on until the end" needs no closing key.
    /// </summary>
    public static void Spans(IReadOnlyList<int> toggles, int clipFrameCount, List<TagSpan> results)
    {
        results.Clear();
        if (toggles == null) return;

        for (var i = 0; i < toggles.Count; i += 2)
        {
            var end = i + 1 < toggles.Count ? toggles[i + 1] : clipFrameCount;
            if (end > toggles[i]) results.Add(new TagSpan(toggles[i], end));
        }
    }

    /// <summary>
    /// The runs of frames within <c>[firstFrame, endFrame)</c> whose active tags satisfy
    /// <paramref name="query"/>, merged where they touch.
    /// </summary>
    /// <remarks>
    /// Evaluates the query once per interval between keys, not per frame. Keys before
    /// <paramref name="firstFrame"/> are not boundaries but still count, so a channel switched on
    /// earlier is already on at the window's start.
    /// </remarks>
    public static void FindSegments(AnnotatedAnimationClip clip, IReadOnlyList<TagChannel> channels,
        GameplayTagQuery query, int firstFrame, int endFrame, List<AnimationClipSegment> results)
    {
        if (results == null) throw new ArgumentNullException(nameof(results));
        if (endFrame <= firstFrame || query == null) return;

        var boundaries = Boundaries(channels, firstFrame, endFrame);
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

        if (runStart >= 0) results.Add(new AnimationClipSegment(clip, runStart, endFrame));
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
    private static List<int> Boundaries(IReadOnlyList<TagChannel> channels, int firstFrame,
        int endFrame)
    {
        var boundaries = new List<int> { firstFrame, endFrame };

        if (channels != null)
        {
            foreach (var channel in channels)
            {
                if (channel?.toggles == null) continue;

                foreach (var toggle in channel.toggles)
                {
                    if (toggle > firstFrame && toggle < endFrame) boundaries.Add(toggle);
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
