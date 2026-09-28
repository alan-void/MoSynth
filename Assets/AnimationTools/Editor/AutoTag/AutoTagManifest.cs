using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GameplayTags;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;
using UnityEditor;

namespace AnimationTools.Editor
{
/// <summary>
/// The run folder's <c>manifest.json</c>, written by the render stage for the Python annotator.
/// </summary>
/// <remarks>
/// Its shape is a contract with the Python <c>autotag</c> package; change both together.
/// </remarks>
public sealed class AutoTagManifest
{
    public const string FileName = "manifest.json";
    public const string VideosFolder = "videos";

    public string Model { get; set; }
    public string MediaResolution { get; set; }
    public float MinConfidence { get; set; }
    public int MinSpanFrames { get; set; }
    public int MinGapFrames { get; set; }

    /// <summary>One model request per tag instead of one per clip; each re-sends the clip's video.</summary>
    public bool PerTagRequests { get; set; }

    public List<TagEntry> Tags { get; set; } = new();
    public List<ClipEntry> Clips { get; set; } = new();

    public sealed class TagEntry
    {
        /// <summary><see cref="GameplayTagSO.TagFullName"/>.</summary>
        public string Name { get; set; }

        public string Description { get; set; }

        /// <summary>Nearest parent first; only those with a description.</summary>
        public List<TagAncestor> Ancestors { get; set; } = new();
    }

    public sealed class TagAncestor
    {
        public string Name { get; set; }
        public string Description { get; set; }
    }

    public sealed class ClipEntry
    {
        public string Guid { get; set; }
        public string Name { get; set; }
        public string AssetPath { get; set; }
        public float ClipFrameRate { get; set; }

        /// <summary>Clip-local, inclusive.</summary>
        public int StartFrame { get; set; }

        /// <summary>Clip-local, exclusive, clamped to the clip's frame count.</summary>
        public int EndFrame { get; set; }

        public float RenderFps { get; set; }
        public int FrameStep { get; set; }

        /// <summary>Relative to the run folder, with forward slashes.</summary>
        public string Video { get; set; }

        /// <summary>Mean ground speed in m/s over each second of the slice.</summary>
        public List<float> GroundSpeed { get; set; } = new();
    }

    /// <summary>The entry for a tag: its full name, description, and described ancestors.</summary>
    public static TagEntry DescribeTag(GameplayTagSO tag)
    {
        var entry = new TagEntry { Name = tag.TagFullName, Description = tag.Description ?? "" };

        for (var ancestor = tag.ParentTag; ancestor != null; ancestor = ancestor.ParentTag)
        {
            if (string.IsNullOrWhiteSpace(ancestor.Description)) continue;

            entry.Ancestors.Add(new TagAncestor { Name = ancestor.TagFullName, Description = ancestor.Description });

            // A reparenting mistake can make a cycle, which TagFullName also caps.
            if (entry.Ancestors.Count > 64) break;
        }

        return entry;
    }

    /// <summary>
    /// The entry for a clip whose video is <paramref name="video"/>. Frame numbers and the ground
    /// speed come from the whole-clip view, since the manifest speaks in clip-local frames.
    /// </summary>
    public static ClipEntry DescribeClip(AnnotatedAnimationClip clip, string video, float renderFps)
    {
        var assetPath = AssetDatabase.GetAssetPath(clip);
        var frameRate = clip.Clip.frameRate;
        var start = clip.startFrame;
        var end = ClipVideoRenderer.SliceEnd(clip);

        return new ClipEntry
        {
            Guid = AssetDatabase.AssetPathToGUID(assetPath),
            Name = clip.name,
            AssetPath = assetPath,
            ClipFrameRate = frameRate,
            StartFrame = start,
            EndFrame = end,
            RenderFps = renderFps,
            FrameStep = ClipVideoRenderer.FrameStep(frameRate, renderFps),
            Video = video,
            GroundSpeed = GroundSpeedPerSecond(
                GaitMeasure.GroundSpeed(clip, clip.Skeleton, start, Math.Max(0, end - start)), frameRate)
        };
    }

    /// <summary>
    /// Means of consecutive one-second windows of a per-frame speed. A trailing part-second is
    /// averaged over the frames it has, so every frame counts towards some entry.
    /// </summary>
    public static List<float> GroundSpeedPerSecond(IReadOnlyList<float> perFrame, float frameRate)
    {
        var result = new List<float>();
        if (perFrame == null || perFrame.Count == 0) return result;

        var window = Math.Max(1, (int)Math.Round(frameRate));
        for (var start = 0; start < perFrame.Count; start += window)
        {
            var end = Math.Min(perFrame.Count, start + window);
            var sum = 0f;
            for (var i = start; i < end; i++) sum += perFrame[i];
            result.Add(sum / (end - start));
        }

        return result;
    }

    public string ToJson() => AutoTagJson.Serialize(this);

    public static AutoTagManifest FromJson(string json) => AutoTagJson.Deserialize<AutoTagManifest>(json);

    public void Write(string runFolder) =>
        File.WriteAllText(Path.Combine(runFolder, FileName), ToJson(), AutoTagJson.Utf8NoBom);

    public static AutoTagManifest Read(string runFolder) =>
        FromJson(File.ReadAllText(Path.Combine(runFolder, FileName), Encoding.UTF8));
}

/// <summary>The run folder's <c>results.json</c>, written by the Python annotator.</summary>
public sealed class AutoTagResults
{
    public const string FileName = "results.json";

    public string Model { get; set; }
    public List<ClipResult> Clips { get; set; } = new();

    public sealed class ClipResult
    {
        public string Guid { get; set; }

        /// <summary>Non-null when the clip failed; nothing is applied for it.</summary>
        public string Error { get; set; }

        public List<TagResult> Tags { get; set; } = new();
    }

    public sealed class TagResult
    {
        public string Name { get; set; }
        public string Reasoning { get; set; }

        /// <summary>Clip-local, half-open, sorted and non-overlapping.</summary>
        public List<Segment> Segments { get; set; } = new();
    }

    public sealed class Segment
    {
        public int StartFrame { get; set; }
        public int EndFrame { get; set; }
        public float Confidence { get; set; }
    }

    public static string PathIn(string runFolder) => Path.Combine(runFolder, FileName);

    public string ToJson() => AutoTagJson.Serialize(this);

    public static AutoTagResults FromJson(string json) => AutoTagJson.Deserialize<AutoTagResults>(json);

    public static AutoTagResults Read(string runFolder) =>
        FromJson(File.ReadAllText(PathIn(runFolder), Encoding.UTF8));
}

/// <summary>The one serializer both run-folder files go through, so their key casing cannot drift.</summary>
internal static class AutoTagJson
{
    public static readonly UTF8Encoding Utf8NoBom = new(false);

    private static readonly JsonSerializerSettings Settings = new()
    {
        ContractResolver = new CamelCasePropertyNamesContractResolver(),
        Formatting = Formatting.Indented,
        NullValueHandling = NullValueHandling.Include,
        Culture = System.Globalization.CultureInfo.InvariantCulture
    };

    public static string Serialize(object value) => JsonConvert.SerializeObject(value, Settings);

    public static T Deserialize<T>(string json) => JsonConvert.DeserializeObject<T>(json, Settings);
}
}
