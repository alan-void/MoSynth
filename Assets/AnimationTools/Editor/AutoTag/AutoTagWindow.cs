using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GameplayTags;
using GameplayTags.Editor;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Tags animation clips with a video model: renders each clip to a video, has
/// <c>python -m autotag.annotate</c> ask the model which tags apply over which frames, and writes
/// the answers into each clip's <see cref="AnimationTagComponent"/>.
/// </summary>
/// <remarks>
/// Each stage reads and writes a run folder under <c>Library/AutoTag/</c>, so any stage can be
/// re-run on its own: re-apply without paying for annotation again, or re-annotate old videos.
/// </remarks>
public sealed class AutoTagWindow : EditorWindow
{
    private const float PricePerMillionInputTokens = 0.75f;
    private const double RepaintInterval = 0.25;

    private enum ClipSource
    {
        Selection,
        Databases
    }

    [SerializeField] private ClipSource clipSource = ClipSource.Selection;
    [SerializeField] private List<ScriptableObject> databases = new();
    [SerializeField] private string selectedRun = "";

    private AutoTagSettingsData _settings;
    private readonly List<AnnotatedAnimationClip> _clips = new();
    private bool _clipsDirty = true;
    private bool _hasApiKey;
    private readonly Dictionary<GameplayTagSO, SerializedObject> _tagObjects = new();

    private AutoTagAnnotateProcess _annotate;
    private bool _applyWhenAnnotated;
    private bool _reloadLocked;
    private double _lastRepaint;

    private string _status;
    private MessageType _statusType = MessageType.Info;
    private Vector2 _scroll;

    [MenuItem("MoSynth/Animation/Auto Tag...")]
    private static void Open()
    {
        var window = GetWindow<AutoTagWindow>("Auto Tag");
        window.minSize = new Vector2(480f, 520f);
        window.Show();
    }

    private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

    /// <summary>Where every run folder lives. Under Library so videos never reach version control.</summary>
    public static string RunsRoot => Path.Combine(ProjectRoot, "Library", "AutoTag");

    private string SelectedRunFolder =>
        string.IsNullOrEmpty(selectedRun) ? null : Path.Combine(RunsRoot, selectedRun);

    private bool IsAnnotating => _annotate != null;

    private void OnEnable()
    {
        _settings = AutoTagSettings.Current.Clone();
        _clipsDirty = true;
        _hasApiKey = AutoTagAnnotateProcess.FindApiKey() != null;
        EditorApplication.update += OnEditorUpdate;
    }

    private void OnDisable()
    {
        EditorApplication.update -= OnEditorUpdate;

        if (_annotate != null)
        {
            Debug.LogWarning("[AutoTag] Window closed; annotation cancelled.");
            _annotate.Dispose();
            _annotate = null;
        }

        UnlockReload();
    }

    /// <summary>Re-reads what may have changed while another window had focus.</summary>
    private void OnFocus()
    {
        _clipsDirty = true;
        _hasApiKey = AutoTagAnnotateProcess.FindApiKey() != null;
    }

    private void OnProjectChange()
    {
        _clipsDirty = true;
        Repaint();
    }

    private void OnSelectionChange()
    {
        if (clipSource != ClipSource.Selection) return;

        _clipsDirty = true;
        Repaint();
    }

    private void OnGUI()
    {
        RefreshClipsIfDirty();

        _scroll = EditorGUILayout.BeginScrollView(_scroll);

        DrawClips();
        EditorGUILayout.Space();
        var tags = ResolveTags();
        DrawTags(tags);
        EditorGUILayout.Space();
        DrawSettings();
        EditorGUILayout.Space();
        DrawEstimate(tags.Count);
        EditorGUILayout.Space();
        DrawRunFolder();
        EditorGUILayout.Space();
        DrawActions(tags);

        if (!string.IsNullOrEmpty(_status)) EditorGUILayout.HelpBox(_status, _statusType);

        EditorGUILayout.EndScrollView();
    }

    private void DrawClips()
    {
        EditorGUILayout.LabelField("Clips", EditorStyles.boldLabel);

        using (var check = new EditorGUI.ChangeCheckScope())
        {
            clipSource = (ClipSource)EditorGUILayout.EnumPopup(
                new GUIContent("Source", "The Project-window selection (folders included), or every " +
                                         "clip of the pose databases listed below."), clipSource);

            if (clipSource == ClipSource.Databases) DrawDatabases();

            if (check.changed) _clipsDirty = true;
        }

        var frames = _clips.Sum(clip => clip.FrameCount);
        EditorGUILayout.LabelField($"{_clips.Count} clips, {frames} frames in their slices",
            EditorStyles.miniLabel);
    }

    private void DrawDatabases()
    {
        for (var i = 0; i < databases.Count; i++)
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                var picked = (ScriptableObject)EditorGUILayout.ObjectField(databases[i], typeof(ScriptableObject), false);
                databases[i] = picked == null || picked is IPoseSetSource ? picked : databases[i];

                if (GUILayout.Button("-", GUILayout.Width(22f)))
                {
                    databases.RemoveAt(i);
                    GUIUtility.ExitGUI();
                }
            }
        }

        var added = (ScriptableObject)EditorGUILayout.ObjectField("Add Database", null, typeof(ScriptableObject), false);
        if (added is IPoseSetSource && !databases.Contains(added)) databases.Add(added);
    }

    private void RefreshClipsIfDirty()
    {
        if (!_clipsDirty) return;

        _clipsDirty = false;
        _clips.Clear();

        IEnumerable<AnnotatedAnimationClip> found = clipSource == ClipSource.Selection
            ? Selection.GetFiltered<AnnotatedAnimationClip>(SelectionMode.Assets | SelectionMode.DeepAssets)
            : databases.OfType<IPoseSetSource>().SelectMany(source => source.AnimationClips ??
                                                                        new List<AnnotatedAnimationClip>());

        foreach (var clip in found)
        {
            if (clip != null && !_clips.Contains(clip)) _clips.Add(clip);
        }
    }

    private List<(string guid, GameplayTagSO tag)> ResolveTags()
    {
        var tags = new List<(string, GameplayTagSO)>();
        foreach (var guid in _settings.tagGuids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var tag = string.IsNullOrEmpty(path) ? null : AssetDatabase.LoadAssetAtPath<GameplayTagSO>(path);
            tags.Add((guid, tag));
        }

        return tags;
    }

    private void DrawTags(List<(string guid, GameplayTagSO tag)> tags)
    {
        EditorGUILayout.LabelField("Tags", EditorStyles.boldLabel);
        EditorGUILayout.LabelField("The model reads each description, so write it for someone who " +
                                   "cannot see the tag tree.", EditorStyles.wordWrappedMiniLabel);

        string removeGuid = null;
        foreach (var (guid, tag) in tags)
        {
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    var label = tag != null ? tag.TagFullName : $"(missing tag {guid})";
                    EditorGUILayout.LabelField(label, EditorStyles.boldLabel);
                    if (GUILayout.Button("Remove", GUILayout.Width(64f))) removeGuid = guid;
                }

                if (tag != null) DrawDescription(tag);
            }
        }

        if (removeGuid != null)
        {
            _settings.tagGuids.Remove(removeGuid);
            SaveSettings();
        }

        var addContent = new GUIContent("Add Tag...");
        var addRect = GUILayoutUtility.GetRect(addContent, GUI.skin.button);
        if (GUI.Button(addRect, addContent)) GameplayTagPicker.ShowWithCreate(addRect, AddTag);

        var undescribed = tags.Where(entry => entry.tag != null && string.IsNullOrWhiteSpace(entry.tag.Description))
            .Select(entry => entry.tag.TagFullName)
            .ToList();
        if (undescribed.Count > 0)
        {
            EditorGUILayout.HelpBox("Give every tag a description before running: " +
                                    string.Join(", ", undescribed), MessageType.Warning);
        }
    }

    private void DrawDescription(GameplayTagSO tag)
    {
        if (!_tagObjects.TryGetValue(tag, out var serialized) || serialized.targetObject == null)
        {
            serialized = new SerializedObject(tag);
            _tagObjects[tag] = serialized;
        }

        serialized.Update();
        var description = serialized.FindProperty("description");
        if (description != null) EditorGUILayout.PropertyField(description, GUIContent.none);
        serialized.ApplyModifiedProperties();

        var context = new List<string>();
        for (var ancestor = tag.ParentTag; ancestor != null && context.Count < 64; ancestor = ancestor.ParentTag)
        {
            if (!string.IsNullOrWhiteSpace(ancestor.Description)) context.Add(ancestor.TagFullName);
        }

        if (context.Count > 0)
        {
            EditorGUILayout.LabelField("Also sends the descriptions of: " + string.Join(", ", context),
                EditorStyles.wordWrappedMiniLabel);
        }
    }

    private void AddTag(GameplayTagSO tag)
    {
        if (tag == null) return;

        var guid = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(tag));
        if (string.IsNullOrEmpty(guid) || _settings.tagGuids.Contains(guid)) return;

        _settings.tagGuids.Add(guid);
        SaveSettings();
        Repaint();
    }

    private void DrawSettings()
    {
        EditorGUILayout.LabelField("Settings", EditorStyles.boldLabel);

        using var check = new EditorGUI.ChangeCheckScope();

        _settings.model = EditorGUILayout.TextField("Model", _settings.model);

        var resolutions = new[] { AutoTagSettings.LowResolution, AutoTagSettings.HighResolution };
        var resolutionIndex = Mathf.Max(0, Array.IndexOf(resolutions, _settings.mediaResolution));
        _settings.mediaResolution = resolutions[EditorGUILayout.Popup(
            new GUIContent("Media Resolution", "Tokens the model spends per video frame: 70 low, 280 high."),
            resolutionIndex, resolutions)];

        _settings.renderFps = EditorGUILayout.FloatField(
            new GUIContent("Render FPS", "Video frames per second. Each video frame skips the nearest whole " +
                                         "number of clip frames."), _settings.renderFps);

        using (new EditorGUILayout.HorizontalScope())
        {
            EditorGUILayout.PrefixLabel("Resolution");
            _settings.width = EditorGUILayout.IntField(_settings.width);
            _settings.height = EditorGUILayout.IntField(_settings.height);
        }

        _settings.minConfidence = EditorGUILayout.Slider(
            new GUIContent("Min Confidence", "Segments the model is less sure of are dropped."),
            _settings.minConfidence, 0f, 1f);
        _settings.minSpanFrames = EditorGUILayout.IntField(
            new GUIContent("Min Span Frames", "Shorter segments are dropped. In clip frames."), _settings.minSpanFrames);
        _settings.minGapFrames = EditorGUILayout.IntField(
            new GUIContent("Min Gap Frames", "Segments closer than this are joined. In clip frames."),
            _settings.minGapFrames);
        _settings.perTagRequests = EditorGUILayout.Toggle(
            new GUIContent("One Request Per Tag", "Ask about each tag separately, so tags cannot influence " +
                                                  "each other. Every request re-sends the video, so input " +
                                                  "cost is multiplied by the number of tags."),
            _settings.perTagRequests);

        if (check.changed) SaveSettings();
    }

    private void SaveSettings()
    {
        AutoTagSettings.Save(_settings);
        _settings = AutoTagSettings.Current.Clone();
    }

    private void DrawEstimate(int tagCount)
    {
        var videoFrames = 0L;
        foreach (var clip in _clips)
        {
            if (!clip.HasClip) continue;

            var step = ClipVideoRenderer.FrameStep(clip.Clip.frameRate, _settings.renderFps);
            videoFrames += ClipVideoRenderer.VideoFrameCount(clip.FrameCount, step);
        }

        var seconds = videoFrames / Math.Max(1f, _settings.renderFps);
        var requestsPerClip = _settings.perTagRequests ? Math.Max(1, tagCount) : 1;
        var tokens = videoFrames * AutoTagSettings.TokensPerFrame(_settings.mediaResolution) * requestsPerClip;
        var dollars = tokens / 1_000_000.0 * PricePerMillionInputTokens;

        EditorGUILayout.LabelField("Estimate", EditorStyles.boldLabel);
        EditorGUILayout.LabelField(
            $"{seconds:0.0} s of video, ~{tokens:N0} input tokens, ~${dollars:0.00} " +
            $"(video frames only, {requestsPerClip} request(s) per clip, " +
            $"at ${PricePerMillionInputTokens}/1M input tokens)",
            EditorStyles.wordWrappedMiniLabel);
    }

    private static string[] ListRuns()
    {
        if (!Directory.Exists(RunsRoot)) return Array.Empty<string>();

        // Run folders are named by timestamp, so the name order is the time order.
        return Directory.GetDirectories(RunsRoot)
            .Select(Path.GetFileName)
            .OrderByDescending(name => name, StringComparer.Ordinal)
            .ToArray();
    }

    private void DrawRunFolder()
    {
        EditorGUILayout.LabelField("Run", EditorStyles.boldLabel);

        var runs = ListRuns();
        using (new EditorGUILayout.HorizontalScope())
        {
            if (runs.Length == 0)
            {
                EditorGUILayout.LabelField("Run Folder", "none yet");
            }
            else
            {
                var index = Mathf.Max(0, Array.IndexOf(runs, selectedRun));
                selectedRun = runs[EditorGUILayout.Popup("Run Folder", index, runs)];
            }

            using (new EditorGUI.DisabledScope(SelectedRunFolder == null || !Directory.Exists(SelectedRunFolder)))
            {
                if (GUILayout.Button("Reveal", GUILayout.Width(64f))) EditorUtility.RevealInFinder(SelectedRunFolder);
            }
        }

        var folder = SelectedRunFolder;
        if (folder == null || !Directory.Exists(folder)) return;

        var hasManifest = File.Exists(Path.Combine(folder, AutoTagManifest.FileName));
        var hasResults = File.Exists(AutoTagResults.PathIn(folder));
        EditorGUILayout.LabelField(
            $"manifest: {(hasManifest ? "yes" : "no")}, results: {(hasResults ? "yes" : "no")}",
            EditorStyles.miniLabel);
    }

    private void DrawActions(List<(string guid, GameplayTagSO tag)> tags)
    {
        var renderBlocker = RenderBlocker(tags);
        var annotateBlocker = AnnotateBlocker();

        if (!_hasApiKey)
        {
            EditorGUILayout.HelpBox($"{AutoTagAnnotateProcess.ApiKeyVariable} is not set, so annotation " +
                                    "cannot run. Set it as a user environment variable.",
                MessageType.Warning);
        }

        if (IsAnnotating)
        {
            DrawAnnotateProgress();
            return;
        }

        var folder = SelectedRunFolder;
        var hasManifest = folder != null && File.Exists(Path.Combine(folder, AutoTagManifest.FileName));
        var hasResults = folder != null && File.Exists(AutoTagResults.PathIn(folder));

        using (new EditorGUILayout.HorizontalScope())
        {
            using (new EditorGUI.DisabledScope(renderBlocker != null))
            {
                if (GUILayout.Button(new GUIContent("Render", renderBlocker), GUILayout.Height(26f))) Render(tags);
            }

            using (new EditorGUI.DisabledScope(annotateBlocker != null || !hasManifest))
            {
                if (GUILayout.Button(new GUIContent("Annotate", annotateBlocker ?? "Needs a rendered run."),
                        GUILayout.Height(26f)))
                {
                    StartAnnotate(folder, applyAfterwards: false);
                }
            }

            using (new EditorGUI.DisabledScope(!hasResults))
            {
                if (GUILayout.Button(new GUIContent("Apply", "Needs an annotated run."), GUILayout.Height(26f)))
                {
                    ApplyRun(folder);
                }
            }
        }

        using (new EditorGUI.DisabledScope(renderBlocker != null || annotateBlocker != null))
        {
            if (GUILayout.Button(new GUIContent("Run All", renderBlocker ?? annotateBlocker), GUILayout.Height(30f)))
            {
                RunAll(tags);
            }
        }

        if (renderBlocker != null) EditorGUILayout.LabelField(renderBlocker, EditorStyles.wordWrappedMiniLabel);
    }

    private string RenderBlocker(List<(string guid, GameplayTagSO tag)> tags)
    {
        if (_clips.Count == 0) return "No clips to render.";
        if (tags.Count == 0) return "No tags to annotate.";
        if (tags.Any(entry => entry.tag == null)) return "A tag in the list is missing; remove it.";
        if (tags.Any(entry => string.IsNullOrWhiteSpace(entry.tag.Description)))
            return "Every tag needs a description.";

        return null;
    }

    private string AnnotateBlocker()
    {
        if (!_hasApiKey)
            return $"{AutoTagAnnotateProcess.ApiKeyVariable} is not set.";

        AutoTagAnnotateProcess.FindPythonExecutable(out var error);
        return error;
    }

    private void DrawAnnotateProgress()
    {
        var label = _annotate.Total > 0
            ? $"Annotating {Math.Min(_annotate.Index + 1, _annotate.Total)} of {_annotate.Total}: {_annotate.CurrentClip}"
            : "Starting annotation...";

        var rect = GUILayoutUtility.GetRect(18f, 20f, GUILayout.ExpandWidth(true));
        EditorGUI.ProgressBar(rect, _annotate.Progress, label);

        EditorGUILayout.LabelField(
            $"Tokens so far: {_annotate.InputTokens:N0} in, {_annotate.OutputTokens:N0} out; " +
            $"{_annotate.ClipErrors} clip errors", EditorStyles.miniLabel);

        if (GUILayout.Button("Cancel"))
        {
            _applyWhenAnnotated = false;
            _annotate.Kill();
        }
    }

    private void RunAll(List<(string guid, GameplayTagSO tag)> tags)
    {
        var folder = Render(tags);
        if (folder != null) StartAnnotate(folder, applyAfterwards: true);
    }

    /// <summary>Renders every clip into a new run folder and writes its manifest. Null on failure or cancel.</summary>
    private string Render(List<(string guid, GameplayTagSO tag)> tags)
    {
        var folder = Path.Combine(RunsRoot, DateTime.Now.ToString("yyyyMMdd-HHmmss"));
        Directory.CreateDirectory(Path.Combine(folder, AutoTagManifest.VideosFolder));

        var manifest = new AutoTagManifest
        {
            Model = _settings.model,
            MediaResolution = _settings.mediaResolution,
            MinConfidence = _settings.minConfidence,
            MinSpanFrames = _settings.minSpanFrames,
            MinGapFrames = _settings.minGapFrames,
            PerTagRequests = _settings.perTagRequests,
            Tags = tags.Select(entry => AutoTagManifest.DescribeTag(entry.tag)).ToList()
        };

        var failed = 0;
        var cancelled = false;
        try
        {
            using var renderer = new ClipVideoRenderer(_settings.width, _settings.height);
            for (var i = 0; i < _clips.Count && !cancelled; i++)
            {
                var clip = _clips[i];
                var guid = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(clip));
                var video = $"{AutoTagManifest.VideosFolder}/{guid}.mp4";

                var index = i;
                var outcome = renderer.Render(clip, Path.Combine(folder, video), _settings.renderFps,
                    out var error, fraction => EditorUtility.DisplayCancelableProgressBar("Render Clips",
                        $"{clip.name} ({index + 1}/{_clips.Count})", (index + fraction) / _clips.Count));

                switch (outcome)
                {
                    case ClipVideoRenderer.Outcome.Rendered:
                        manifest.Clips.Add(AutoTagManifest.DescribeClip(clip, video, _settings.renderFps));
                        break;

                    case ClipVideoRenderer.Outcome.Cancelled:
                        cancelled = true;
                        break;

                    default:
                        failed++;
                        Debug.LogError($"[AutoTag] Could not render {clip.name}: {error}", clip);
                        break;
                }
            }
        }
        finally
        {
            EditorUtility.ClearProgressBar();
        }

        if (cancelled || manifest.Clips.Count == 0)
        {
            Directory.Delete(folder, true);
            SetStatus(cancelled ? "Render cancelled." : "No clip rendered; see the Console.",
                cancelled ? MessageType.Info : MessageType.Error);
            return null;
        }

        manifest.Write(folder);
        selectedRun = Path.GetFileName(folder);
        SetStatus($"Rendered {manifest.Clips.Count} clips to {folder}" +
                  (failed > 0 ? $"; {failed} failed, see the Console." : "."),
            failed > 0 ? MessageType.Warning : MessageType.Info);
        return folder;
    }

    private void StartAnnotate(string folder, bool applyAfterwards)
    {
        _annotate = AutoTagAnnotateProcess.Start(folder, out var error);
        if (_annotate == null)
        {
            SetStatus(error, MessageType.Error);
            return;
        }

        _applyWhenAnnotated = applyAfterwards;

        // A domain reload would orphan the process and lose its output.
        EditorApplication.LockReloadAssemblies();
        _reloadLocked = true;

        SetStatus("Annotating...", MessageType.Info);
    }

    private void OnEditorUpdate()
    {
        if (_annotate == null) return;

        if (_annotate.Update())
        {
            FinishAnnotate();
            return;
        }

        if (EditorApplication.timeSinceStartup - _lastRepaint < RepaintInterval) return;

        _lastRepaint = EditorApplication.timeSinceStartup;
        Repaint();
    }

    private void FinishAnnotate()
    {
        var process = _annotate;
        _annotate = null;
        UnlockReload();

        var exitCode = process.ExitCode;
        var succeeded = exitCode is AutoTagAnnotateProcess.ExitSuccess or AutoTagAnnotateProcess.ExitPartial;
        var tokens = $"{process.InputTokens:N0} input and {process.OutputTokens:N0} output tokens";

        if (!succeeded)
        {
            Debug.LogError($"[AutoTag] Annotation failed (exit code {exitCode}).\n{process.Stderr}");
            SetStatus($"Annotation failed (exit code {exitCode}); see the Console.", MessageType.Error);
        }
        else
        {
            SetStatus(exitCode == AutoTagAnnotateProcess.ExitPartial
                    ? $"Annotated with {process.ClipErrors} clip errors, using {tokens}; see the Console."
                    : $"Annotated, using {tokens}.",
                exitCode == AutoTagAnnotateProcess.ExitPartial ? MessageType.Warning : MessageType.Info);
        }

        var apply = succeeded && _applyWhenAnnotated;
        _applyWhenAnnotated = false;
        process.Dispose();

        if (apply) ApplyRun(process.RunFolder);
        Repaint();
    }

    private void UnlockReload()
    {
        if (!_reloadLocked) return;

        EditorApplication.UnlockReloadAssemblies();
        _reloadLocked = false;
    }

    private void ApplyRun(string folder)
    {
        AutoTagResults results;
        try
        {
            results = AutoTagResults.Read(folder);
        }
        catch (Exception e)
        {
            SetStatus($"Could not read {AutoTagResults.PathIn(folder)}: {e.Message}", MessageType.Error);
            return;
        }

        var overwrite = false;
        var existing = AutoTagApply.CountExistingChannels(results);
        if (existing > 0)
        {
            var choice = EditorUtility.DisplayDialogComplex("Apply Auto Tags",
                $"{existing} tag channels already hold keys, which may have been set by hand. " +
                "Overwriting replaces them.",
                "Skip Existing", "Cancel", "Overwrite");

            switch (choice)
            {
                case 0: break;
                case 2: overwrite = true; break;
                default: return;
            }
        }

        var report = AutoTagApply.ApplyAll(results, overwrite);
        Debug.Log($"[AutoTag] Apply: {report.Summary()}");
        SetStatus(report.Summary(), MessageType.Info);
        EditorUtility.DisplayDialog("Apply Auto Tags", report.Summary(), "OK");
    }

    private void SetStatus(string status, MessageType type)
    {
        _status = status;
        _statusType = type;
    }
}
}
