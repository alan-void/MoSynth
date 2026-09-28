using System.Collections.Generic;
using System.Linq;
using GameplayTags;
using GameplayTags.Editor;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Writes an auto-tag run's <c>results.json</c> into each clip's <see cref="AnimationTagComponent"/>,
/// one channel per tag.
/// </summary>
/// <remarks>
/// Nothing records that a channel was keyed by hand, so a channel that already holds keys is skipped
/// unless the caller asks to overwrite; that choice is the only guard.
/// </remarks>
public static class AutoTagApply
{
    /// <summary>What an apply did, for the console and the closing dialog.</summary>
    public sealed class Report
    {
        public int ClipsWritten;
        public int ChannelsWritten;
        public int ChannelsSkipped;     // already held keys and overwrite was off
        public int ClipsFailed;         // the annotator reported an error for them
        public int ClipsMissing;        // no clip asset at the GUID
        public int UnknownTags;         // tag names with no tag asset
        public bool Cancelled;

        public string Summary()
        {
            var parts = new List<string>();
            if (ClipsWritten > 0) parts.Add($"{ChannelsWritten} channels written across {ClipsWritten} clips");
            if (ChannelsSkipped > 0) parts.Add($"{ChannelsSkipped} channels skipped (already had keys)");
            if (ClipsFailed > 0) parts.Add($"{ClipsFailed} clips skipped (annotation failed)");
            if (ClipsMissing > 0) parts.Add($"{ClipsMissing} clips not found");
            if (UnknownTags > 0) parts.Add($"{UnknownTags} tag results naming no known tag");

            var summary = parts.Count == 0 ? "Nothing to do" : string.Join(", ", parts);
            return Cancelled ? summary + ". Cancelled part way; what had run was kept." : summary + ".";
        }
    }

    /// <summary>
    /// Channel keys for a set of half-open segments: <c>[s0, e0, s1, e1, ...]</c>. Sorts, drops empty
    /// segments, and merges ones that overlap or touch, since two keys on one frame would cancel.
    /// </summary>
    public static List<int> SegmentsToToggles(IEnumerable<(int start, int end)> segments)
    {
        var toggles = new List<int>();
        if (segments == null) return toggles;

        var sorted = segments.Where(segment => segment.end > segment.start)
            .OrderBy(segment => segment.start)
            .ThenBy(segment => segment.end)
            .ToList();

        for (var i = 0; i < sorted.Count;)
        {
            var start = sorted[i].start;
            var end = sorted[i].end;

            for (i++; i < sorted.Count && sorted[i].start <= end; i++)
            {
                if (sorted[i].end > end) end = sorted[i].end;
            }

            toggles.Add(start);
            toggles.Add(end);
        }

        return toggles;
    }

    /// <summary>
    /// How many (clip, tag) results would land on a channel that already holds keys; what the caller
    /// asks the user about before choosing to overwrite.
    /// </summary>
    public static int CountExistingChannels(AutoTagResults results)
    {
        var count = 0;
        foreach (var clipResult in Applicable(results))
        {
            var clip = LoadClip(clipResult.Guid);
            if (clip == null || !clip.TryGetComponent<AnimationTagComponent>(out var component)) continue;

            foreach (var tagResult in clipResult.Tags)
            {
                var channel = FindChannel(component, ResolveTag(tagResult.Name));
                if (channel != null && channel.toggles.Count > 0) count++;
            }
        }

        return count;
    }

    public static Report ApplyAll(AutoTagResults results, bool overwrite)
    {
        var report = new Report();
        if (results?.Clips == null) return report;

        var undoGroup = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("Apply auto tags");

        AssetDatabase.StartAssetEditing();
        try
        {
            for (var i = 0; i < results.Clips.Count; i++)
            {
                var clipResult = results.Clips[i];
                if (clipResult == null) continue;

                if (clipResult.Error != null)
                {
                    report.ClipsFailed++;
                    continue;
                }

                var clip = LoadClip(clipResult.Guid);
                if (clip == null)
                {
                    report.ClipsMissing++;
                    Debug.LogWarning($"[AutoTag] No annotated clip with GUID {clipResult.Guid}.");
                    continue;
                }

                if (EditorUtility.DisplayCancelableProgressBar("Apply Auto Tags", clip.name,
                        (float)i / results.Clips.Count))
                {
                    report.Cancelled = true;
                    break;
                }

                Apply(clip, clipResult, overwrite, report);
            }
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            AssetDatabase.StopAssetEditing();
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        Undo.CollapseUndoOperations(undoGroup);

        // Written straight to the asset, so an open editor is still showing the channels from before.
        AnnotatedClipEditorWindow.RefreshOpenWindows();
        return report;
    }

    private static void Apply(AnnotatedAnimationClip clip, AutoTagResults.ClipResult clipResult, bool overwrite,
        Report report)
    {
        // A managed-reference list edit is not tracked reliably by RecordObject's diffing, so the
        // whole object is snapshotted. That does not dirty the asset.
        Undo.RegisterCompleteObjectUndo(clip, "Apply auto tags");

        var seeded = !clip.TryGetComponent<AnimationTagComponent>(out var component);
        if (seeded)
        {
            component = new AnimationTagComponent();
            clip.components.Add(component);
        }

        var written = 0;
        foreach (var tagResult in clipResult.Tags ?? new List<AutoTagResults.TagResult>())
        {
            var tag = ResolveTag(tagResult?.Name);
            if (tag == null)
            {
                report.UnknownTags++;
                Debug.LogWarning($"[AutoTag] {clip.name}: no tag named \"{tagResult?.Name}\".", clip);
                continue;
            }

            var toggles = SegmentsToToggles((tagResult.Segments ?? new List<AutoTagResults.Segment>())
                .Select(segment => (segment.StartFrame, segment.EndFrame)));

            var channel = FindChannel(component, tag);
            if (channel != null && channel.toggles.Count > 0 && !overwrite)
            {
                report.ChannelsSkipped++;
                continue;
            }

            // A tag that never applies gets no new channel; there is nothing to show on it.
            if (channel == null && toggles.Count == 0) continue;

            if (channel == null)
            {
                channel = new AnimationTagging.TagChannel { tag = tag };
                component.channels.Add(channel);
            }

            channel.toggles.Clear();
            channel.toggles.AddRange(toggles);
            AnimationTagging.Normalise(channel.toggles);
            written++;
        }

        if (written == 0)
        {
            // Do not leave the clip with an empty component it did not have before.
            if (seeded) clip.components.Remove(component);
            return;
        }

        report.ClipsWritten++;
        report.ChannelsWritten += written;
        EditorUtility.SetDirty(clip);
    }

    private static IEnumerable<AutoTagResults.ClipResult> Applicable(AutoTagResults results) =>
        results?.Clips == null
            ? Enumerable.Empty<AutoTagResults.ClipResult>()
            : results.Clips.Where(clip => clip != null && clip.Error == null && clip.Tags != null);

    private static AnnotatedAnimationClip LoadClip(string guid)
    {
        if (string.IsNullOrEmpty(guid)) return null;

        var path = AssetDatabase.GUIDToAssetPath(guid);
        return string.IsNullOrEmpty(path) ? null : AssetDatabase.LoadAssetAtPath<AnnotatedAnimationClip>(path);
    }

    private static GameplayTagSO ResolveTag(string fullName) =>
        string.IsNullOrEmpty(fullName) ? null : GameplayTagConfig.instance.GetTag(fullName);

    private static AnimationTagging.TagChannel FindChannel(AnimationTagComponent component, GameplayTagSO tag)
    {
        if (component?.channels == null || tag == null) return null;

        foreach (var channel in component.channels)
        {
            if (channel != null && channel.tag == tag) return channel;
        }

        return null;
    }
}
}
