using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Creates an <see cref="AnnotatedAnimationClip"/> for every take in one imported model that matches
/// a name filter, into a folder of your choosing.
/// </summary>
/// <remarks>
/// The single-selection menu takes one clip at a time and writes beside the source model, which is
/// the wrong shape for a retargeted capture session: those arrive as hundreds of takes in one FBX,
/// under a source tree that is not where the assets belong. See
/// <c>openwiki/animation-tools/animation-sources.md</c>.
/// </remarks>
public sealed class BatchAnnotatedClipWindow : EditorWindow
{
    [SerializeField] private GameObject model;
    [SerializeField] private string nameFilter = "";
    [SerializeField] private string outputFolder = "Assets/Animation";
    [SerializeField] private bool addGaitPhase = true;
    [SerializeField] private bool detectFootfalls = true;

    private readonly List<AnimationClip> _matches = new();
    private string _status;
    private MessageType _statusType = MessageType.Info;
    private Vector2 _scroll;

    [MenuItem("MoSynth/Animation/Create Annotated Clips From Model...")]
    private static void Open()
    {
        var window = GetWindow<BatchAnnotatedClipWindow>(true, "Create Annotated Clips");
        window.minSize = new Vector2(460f, 380f);
        window.Show();
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("Source", EditorStyles.boldLabel);

        using (var check = new EditorGUI.ChangeCheckScope())
        {
            model = (GameObject)EditorGUILayout.ObjectField("Model", model, typeof(GameObject), false);
            nameFilter = EditorGUILayout.TextField(
                new GUIContent("Name Filter", "Case-insensitive substring of the take name. Empty takes every clip."),
                nameFilter);

            if (check.changed)
            {
                _matches.Clear();
                _status = null;
            }
        }

        DrawOutputFolder();

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Annotation", EditorStyles.boldLabel);
        addGaitPhase = EditorGUILayout.Toggle(
            new GUIContent("Add Gait Phase", "Adds a GaitPhaseComponent to clips that have none. Existing components are left alone."),
            addGaitPhase);

        using (new EditorGUI.DisabledScope(!addGaitPhase))
        {
            detectFootfalls = EditorGUILayout.Toggle(
                new GUIContent("Detect Footfalls", "Runs detection on the components this pass adds."),
                detectFootfalls);
        }

        EditorGUILayout.Space();

        using (new EditorGUI.DisabledScope(model == null))
        {
            if (GUILayout.Button("Preview", GUILayout.Height(22f))) Preview();
        }

        DrawMatches();

        if (!string.IsNullOrEmpty(_status)) EditorGUILayout.HelpBox(_status, _statusType);

        using (new EditorGUI.DisabledScope(_matches.Count == 0 || !IsOutputFolderValid()))
        {
            if (GUILayout.Button($"Create {_matches.Count} Annotated Clip(s)", GUILayout.Height(28f)))
            {
                Create();
            }
        }
    }

    private void DrawOutputFolder()
    {
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Output", EditorStyles.boldLabel);

        using (new EditorGUILayout.HorizontalScope())
        {
            outputFolder = EditorGUILayout.TextField("Folder", outputFolder);
            if (GUILayout.Button("Browse...", GUILayout.Width(80f)))
            {
                var picked = EditorUtility.OpenFolderPanel("Output folder", outputFolder, "");
                if (!string.IsNullOrEmpty(picked)) outputFolder = ToProjectRelative(picked);
            }
        }

        if (!IsOutputFolderValid())
        {
            EditorGUILayout.HelpBox("Pick a folder inside this project's Assets.", MessageType.Warning);
        }
    }

    private void DrawMatches()
    {
        if (_matches.Count == 0) return;

        _scroll = EditorGUILayout.BeginScrollView(_scroll, GUILayout.MinHeight(120f));
        foreach (var clip in _matches)
        {
            EditorGUILayout.LabelField(clip.name, $"{SkeletonAnimation.FrameCountOf(clip)} frames");
        }

        EditorGUILayout.EndScrollView();
    }

    private static string ToProjectRelative(string absolutePath)
    {
        var normalized = absolutePath.Replace('\\', '/');
        var projectRoot = Directory.GetParent(Application.dataPath)?.FullName.Replace('\\', '/');
        if (projectRoot != null && normalized.StartsWith(projectRoot + "/"))
        {
            return normalized.Substring(projectRoot.Length + 1);
        }

        return normalized;
    }

    private bool IsOutputFolderValid()
    {
        return !string.IsNullOrWhiteSpace(outputFolder) &&
               outputFolder.Replace('\\', '/').StartsWith("Assets/");
    }

    private void Preview()
    {
        _matches.Clear();

        if (!AnnotatedClipFactory.TryResolveModel(model, out _, out var clips, out var error))
        {
            SetStatus(error, MessageType.Error);
            return;
        }

        var total = 0;
        foreach (var clip in clips)
        {
            if (!AnnotatedClipFactory.MatchesFilter(clip.name, nameFilter)) continue;

            _matches.Add(clip);
            total += SkeletonAnimation.FrameCountOf(clip);
        }

        var assetPath = AssetDatabase.GetAssetPath(model);
        if (_matches.Count == 0)
        {
            SetStatus($"No take in '{Path.GetFileName(assetPath)}' matches \"{nameFilter}\" ({clips.Count} takes in the file).",
                MessageType.Warning);
            return;
        }

        var reduction = AnnotatedClipFactory.UsesKeyframeReduction(assetPath)
            ? " This model still uses keyframe reduction; setting Anim. Compression to Off improves bake fidelity."
            : "";

        SetStatus($"{_matches.Count} of {clips.Count} takes match, {total} frames total.{reduction}",
            string.IsNullOrEmpty(reduction) ? MessageType.Info : MessageType.Warning);
    }

    private void Create()
    {
        var report = AnnotatedClipFactory.CreateBatch(model, nameFilter, outputFolder, addGaitPhase,
            detectFootfalls);

        SetStatus(report.Summary(),
            !report.Ok || report.DetectionFailed > 0 ? MessageType.Warning : MessageType.Info);

        if (report.Ok) Debug.Log($"Annotated clips in '{outputFolder}': {report.Summary()}");
    }

    private void SetStatus(string message, MessageType type)
    {
        _status = message;
        _statusType = type;
    }
}
}
