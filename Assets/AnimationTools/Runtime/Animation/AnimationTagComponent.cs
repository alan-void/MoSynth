using System;
using System.Collections.Generic;
using GameplayTags;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// What is true about a clip over which stretches of it — a tag per channel, switched on and off by
/// keyframes.
/// </summary>
/// <remarks>
/// Tags are hierarchical (<c>action.walk</c>, <c>style.tired</c>), so a question asked about
/// <c>action</c> is answered by every clip annotated below it and the vocabulary can be refined
/// without re-annotating what already exists.
/// <para>
/// This is the author the <c>.mmpose</c> tag block never had. It does not feed that block yet:
/// nothing reads tags out of the pose database, so writing them there would only be dead weight
/// with a database regeneration attached. See <c>openwiki/animation-tools/clip-tags.md</c>.
/// </para>
/// </remarks>
[Serializable]
public sealed class AnimationTagComponent : AnimationClipComponent
{
    [Tooltip("One tag each, and the frames it switches on and off at.")]
    public List<AnimationTagging.TagChannel> channels = new();

    public override string Describe()
    {
        if (channels == null || channels.Count == 0) return "no tag channels";

        var keys = 0;
        var unnamed = 0;
        foreach (var channel in channels)
        {
            if (channel == null) continue;
            if (channel.tag == null) unnamed++;

            keys += channel.toggles?.Count ?? 0;
        }

        var summary = $"{channels.Count} channels, {keys} keys";
        return unnamed == 0 ? summary : $"{summary}, {unnamed} with no tag";
    }

    /// <summary>Normalises every channel and merges channels that name the same tag.</summary>
    /// <remarks>
    /// Like <see cref="GaitPhaseComponent.OnValidate"/> this silently drops keys the clip's frame
    /// range no longer contains, which is why the editor reports what it lost rather than leaving
    /// you to notice.
    /// </remarks>
    public override void OnValidate(AnnotatedAnimationClip clip)
    {
        if (channels == null || clip == null) return;

        channels.RemoveAll(channel => channel == null);

        // Two channels for one tag would each answer for it, and a query would see whichever came
        // first. Folding them is the only reading that keeps a channel the single truth about a tag.
        for (var i = 0; i < channels.Count; i++)
        {
            for (var j = channels.Count - 1; j > i; j--)
            {
                if (channels[i].tag == null || channels[j].tag != channels[i].tag) continue;

                channels[i].toggles.AddRange(channels[j].toggles);
                channels.RemoveAt(j);
            }
        }

        var frameCount = clip.FrameCount;
        foreach (var channel in channels)
        {
            AnimationTagging.Normalise(channel.toggles, frameCount);
        }
    }

    /// <summary>Whether <paramref name="tag"/>'s own channel is on at a sliced frame.</summary>
    /// <remarks>
    /// Exact, not hierarchical: this asks about one channel. Ask a hierarchical question with
    /// <see cref="TagsAt"/> and a <see cref="GameplayTagQuery"/>.
    /// </remarks>
    public bool IsOn(GameplayTagSO tag, int sliceFrame)
    {
        if (channels == null || tag == null) return false;

        foreach (var channel in channels)
        {
            if (channel?.tag == tag) return AnimationTagging.IsOn(channel.toggles, sliceFrame);
        }

        return false;
    }

    /// <summary>The tags on at a sliced frame, replacing whatever <paramref name="into"/> held.</summary>
    public void TagsAt(int sliceFrame, GameplayTagSet into) =>
        AnimationTagging.CollectTagsAt(channels, sliceFrame, into);

    /// <summary>Appends the runs of <paramref name="clip"/> whose tags satisfy the query.</summary>
    public void FindSegments(AnnotatedAnimationClip clip, GameplayTagQuery query,
        List<AnimationClipSegment> results)
    {
        if (clip == null) return;

        AnimationTagging.FindSegments(clip, channels, query, clip.FrameCount, results);
    }

    /// <summary>
    /// Appends every matching run across a set of clips, skipping those with no tag component and
    /// those whose component is disabled.
    /// </summary>
    public static void FindSegments(IEnumerable<AnnotatedAnimationClip> clips,
        GameplayTagQuery query, List<AnimationClipSegment> results)
    {
        if (clips == null) return;

        foreach (var clip in clips)
        {
            if (clip == null || !clip.TryGetComponent<AnimationTagComponent>(out var tags)) continue;
            if (!tags.isEnabled) continue;

            tags.FindSegments(clip, query, results);
        }
    }
}
}
