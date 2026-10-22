using System;
using System.IO;
using AnimationTools;
using AnimationTools.Editor;
using UnityEditor;
using UnityEngine;

namespace Style.Editor
{
/// <summary>
/// Inspector for <see cref="StyleTransferConfig"/>: bake the labelled pose database, choose which
/// bones are restyled, then train.
/// </summary>
[CustomEditor(typeof(StyleTransferConfig))]
public class StyleTransferConfigEditor : UnityEditor.Editor
{
    private bool _showBones = true;

    // Caches the skeleton, because OnInspectorGUI repaints constantly; dropped on enable and on
    // regeneration.
    private readonly BoneSelectionTreeGUI _boneTree = new("styled", "Style All");

    private SerializedProperty _contactBonesProperty;

    private void OnEnable()
    {
        _boneTree.Invalidate();
        _contactBonesProperty = serializedObject.FindProperty("contactBones");
    }

    public override void OnInspectorGUI()
    {
        var config = (StyleTransferConfig)target;

        serializedObject.Update();

        // Scoped to the fields: GUI.changed is also set by a button press, so a blanket check would
        // wipe hasTrained the instant Train set it.
        EditorGUI.BeginChangeCheck();
        DrawPropertiesExcluding(serializedObject, "m_Script", "contactBones");
        PoseSetSourceGUI.DrawContactBones(_contactBonesProperty, config.Skeleton);
        var fieldsChanged = EditorGUI.EndChangeCheck();

        serializedObject.ApplyModifiedProperties();

        // Which field moved is not knowable here, so every edit invalidates both artefacts.
        if (fieldsChanged) MarkStale(config, database: true);

        // The clips themselves are checked when Generate is pressed: checking them here would load
        // every clip for as long as the asset is inspected.
        var settingsValid = config.TryValidateSettings(out var settingsError);
        if (!settingsValid) EditorGUILayout.HelpBox(settingsError, MessageType.Error);

        EditorGUILayout.Space();
        DrawDatabaseSection(config, settingsValid);

        EditorGUILayout.Space();
        DrawBoneSection(config);

        EditorGUILayout.Space();
        DrawTrainingSection(config);

        if (GUI.changed) EditorUtility.SetDirty(config);
    }

    /// <summary>
    /// Note that an artefact no longer matches the config. <paramref name="database"/> extends that
    /// to the database, which drags the checkpoint with it.
    /// </summary>
    private static void MarkStale(StyleTransferConfig config, bool database)
    {
        if (database) config.hasPoseDatabase = false;
        config.hasTrained = false;
        EditorUtility.SetDirty(config);
    }

    private void DrawDatabaseSection(StyleTransferConfig config, bool settingsValid)
    {
        EditorGUILayout.LabelField("Pose Database", EditorStyles.boldLabel);
        EditorGUILayout.LabelField("Output", ProjectRelative(config.GetAssetPath()));

        if (File.Exists(config.GetPoseDatabasePath()))
        {
            DrawBuildState(config.hasPoseDatabase,
                "The config changed since the pose database was generated. Regenerate it, then " +
                "retrain.");
        }

        using (new EditorGUI.DisabledScope(!settingsValid))
        {
            if (GUILayout.Button("Generate Pose Database", GUILayout.Height(24)))
            {
                GeneratePoseDatabase(config);
                _boneTree.Invalidate();
            }
        }
    }

    private static void DrawBuildState(bool current, string staleMessage)
    {
        if (current)
        {
            EditorGUILayout.LabelField(" ", "Up to date with this config.", EditorStyles.miniLabel);
            return;
        }

        EditorGUILayout.HelpBox(staleMessage, MessageType.Error);
    }

    // --- Styled bones ---------------------------------------------------------------------------

    private void DrawBoneSection(StyleTransferConfig config)
    {
        _showBones = EditorGUILayout.Foldout(_showBones, "Styled Bones", true);
        if (!_showBones) return;

        if (!_boneTree.TryRead(config.TryGetDatabaseSkeleton))
        {
            EditorGUILayout.HelpBox("Assign this config's skeleton first.", MessageType.Warning);
            return;
        }

        EditorGUILayout.HelpBox(
            "Bones the model restyles. Unticked bones keep the input pose's rotation.\n" +
            "Unticking a bone unticks its whole subtree. The root bone must stay styled.\n" +
            "Changing the selection changes the model's shape, so it needs a retrain.",
            MessageType.None);

        // The extraction does not change, only the model's shape.
        if (_boneTree.Draw(config, config.excludedBones)) MarkStale(config, database: false);
    }

    // --- Training -------------------------------------------------------------------------------

    private void DrawTrainingSection(StyleTransferConfig config)
    {
        EditorGUILayout.LabelField("Model", EditorStyles.boldLabel);

        var checkpointPath = config.GetCheckpointPath();
        var databaseExists = File.Exists(config.GetPoseDatabasePath()) &&
                             File.Exists(config.GetLabelsPath());
        var trained = File.Exists(checkpointPath);

        EditorGUILayout.LabelField("Checkpoint",
            trained
                ? $"{ProjectRelative(checkpointPath)}  ({new FileInfo(checkpointPath).Length / 1024} KB)"
                : "not trained yet");

        if (!databaseExists)
        {
            EditorGUILayout.HelpBox("Generate the pose database first.", MessageType.Warning);
        }
        else if (trained)
        {
            DrawBuildState(config.hasTrained,
                "The config changed since the model was trained. Until it is retrained the stage " +
                "may refuse the checkpoint, or run a model shaped for a different config.");
        }

        if (EditorApplication.isPlayingOrWillChangePlaymode)
        {
            EditorGUILayout.HelpBox(
                "Training is disabled in play mode. It would hold the Python GIL for as long as it " +
                "runs and stall the StyleTransferStage.", MessageType.Info);
        }

        using (new EditorGUI.DisabledScope(
                   !databaseExists || EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button("Train Style Model", GUILayout.Height(30))) StyleTraining.Run(config);
        }
    }

    /// <summary>
    /// Extract and serialize the pose database and its style labels, and mark any trained model
    /// stale. Returns false if they did not get written.
    /// </summary>
    public static bool GeneratePoseDatabase(StyleTransferConfig config)
    {
        if (!config.TryValidate(out var error) ||
            !config.TryBuildClipLabels(out var labels, out error))
        {
            Debug.LogError($"[Style] '{config.name}': {error}", config);
            return false;
        }

        try
        {
            EditorUtility.DisplayProgressBar("Style", "Extracting poses...", 0.3f);
            config.ImportPoseSet();
            var poseSet = config.GetOrImportPoseSet();
            if (poseSet == null) return false;

            // The labels are matched to database clips by position alone, so a count that differs
            // would silently shift every label after the gap.
            if (poseSet.NumberClips != labels.Count)
            {
                Debug.LogError($"[Style] '{config.name}': the database holds {poseSet.NumberClips} " +
                               $"clips but {labels.Count} were labelled; nothing was written.", config);
                return false;
            }

            EditorUtility.DisplayProgressBar("Style", "Writing database...", 0.7f);
            new PoseSerializer().Serialize(poseSet, config.GetAssetPath(), config.name);
            File.WriteAllText(config.GetLabelsPath(), StyleTransferConfig.LabelsToJson(labels));

            Debug.Log($"[Style] Wrote {poseSet.NumberPoses} poses and {labels.Count} clip labels to " +
                      $"{ProjectRelative(config.GetAssetPath())}.");

            config.hasPoseDatabase = true;
            config.hasTrained = false;
            EditorUtility.SetDirty(config);
            AssetDatabase.SaveAssetIfDirty(config);
            return true;
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            return false;
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            config.InvalidatePoseSet();
            AssetDatabase.Refresh();
        }
    }

    private static string ProjectRelative(string absolutePath)
    {
        var project = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        var full = Path.GetFullPath(absolutePath);
        return full.StartsWith(project, StringComparison.OrdinalIgnoreCase)
            ? full[(project.Length + 1)..]
            : full;
    }
}
}
