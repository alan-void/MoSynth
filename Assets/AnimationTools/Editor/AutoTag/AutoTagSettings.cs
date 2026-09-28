using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>What an auto-tag run asks for: the model, how clips are rendered, and which tags.</summary>
[Serializable]
public sealed class AutoTagSettingsData
{
    public string model = "gemini-3.8-flash";

    [Tooltip("\"low\" or \"high\": how many tokens the model spends per video frame.")]
    public string mediaResolution = AutoTagSettings.LowResolution;

    public float renderFps = 10f;
    public int width = 640;
    public int height = 480;
    public float minConfidence = 0.5f;
    public int minSpanFrames = 3;
    public int minGapFrames = 3;

    [Tooltip("Ask about each tag in its own request, so tags cannot influence each other. " +
             "Multiplies the video's input cost by the number of tags.")]
    public bool perTagRequests;

    [Tooltip("Asset GUIDs of the tags to annotate, so renaming a tag does not drop it.")]
    public List<string> tagGuids = new();

    public AutoTagSettingsData Clone()
    {
        var copy = (AutoTagSettingsData)MemberwiseClone();
        copy.tagGuids = new List<string>(tagGuids ?? new List<string>());
        return copy;
    }
}

/// <summary>
/// Per-user store for <see cref="AutoTagSettingsData"/>, in gitignored <c>UserSettings/</c>, since
/// which model to pay for and what to tag with are one person's choices, not the project's.
/// </summary>
public static class AutoTagSettings
{
    public const string ProjectRelativePath = "UserSettings/MoSynthAutoTag.json";
    public const string LowResolution = "low";
    public const string HighResolution = "high";

    private static AutoTagSettingsData _cached;

    public static string FilePath =>
        Path.GetFullPath(Path.Combine(Application.dataPath, "..", ProjectRelativePath));

    /// <summary>The stored settings, read once per domain; defaults when the file is missing or unreadable.</summary>
    public static AutoTagSettingsData Current => _cached ??= ReadFile();

    public static void Save(AutoTagSettingsData settings)
    {
        var normalised = Normalise(settings);
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath) ?? "");
        File.WriteAllText(FilePath, JsonUtility.ToJson(normalised, true));
        _cached = normalised;
    }

    /// <summary>Drops the cache, so the next read picks up a file edited outside the Editor.</summary>
    public static void Reload() => _cached = null;

    /// <summary>Reads the JSON form; anything unparseable reads as the defaults.</summary>
    public static AutoTagSettingsData Parse(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return new AutoTagSettingsData();

        try
        {
            return Normalise(JsonUtility.FromJson<AutoTagSettingsData>(json));
        }
        catch (ArgumentException)
        {
            return new AutoTagSettingsData();
        }
    }

    /// <summary>Clamps values a hand-edited file could leave out of range.</summary>
    public static AutoTagSettingsData Normalise(AutoTagSettingsData settings)
    {
        var result = settings?.Clone() ?? new AutoTagSettingsData();

        if (string.IsNullOrWhiteSpace(result.model)) result.model = new AutoTagSettingsData().model;
        if (result.mediaResolution != HighResolution) result.mediaResolution = LowResolution;

        result.renderFps = Mathf.Clamp(result.renderFps, 1f, 60f);
        result.width = Mathf.Clamp(result.width, 64, 3840);
        result.height = Mathf.Clamp(result.height, 64, 2160);
        result.minConfidence = Mathf.Clamp01(result.minConfidence);
        result.minSpanFrames = Mathf.Max(1, result.minSpanFrames);
        result.minGapFrames = Mathf.Max(0, result.minGapFrames);
        result.tagGuids.RemoveAll(string.IsNullOrEmpty);
        return result;
    }

    /// <summary>Input tokens the model spends on one video frame at a media resolution.</summary>
    public static int TokensPerFrame(string mediaResolution) =>
        mediaResolution == HighResolution ? 280 : 70;

    private static AutoTagSettingsData ReadFile()
    {
        var path = FilePath;
        if (!File.Exists(path)) return new AutoTagSettingsData();

        try
        {
            return Parse(File.ReadAllText(path));
        }
        catch (IOException e)
        {
            Debug.LogWarning($"Could not read \"{path}\" ({e.Message}); using default auto-tag settings.");
            return new AutoTagSettingsData();
        }
    }
}
}
