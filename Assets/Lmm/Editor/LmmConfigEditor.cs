using System;
using System.IO;
using AnimationTools;
using AnimationTools.Editor;
using UnityEditor;
using UnityEngine;

namespace Lmm.Editor
{
/// <summary>
/// Inspector for <see cref="LmmConfig"/>: point it at a database, choose which bones the
/// decompressor predicts, then train.
/// </summary>
[CustomEditor(typeof(LmmConfig))]
public class LmmConfigEditor : UnityEditor.Editor
{
    private bool _showBones = true;

    // Caches the assigned database's skeleton, because OnInspectorGUI repaints constantly.
    private readonly BoneSelectionTreeGUI _boneTree = new("predicted", "Predict All");

    private void OnEnable() => InvalidateSkeleton();

    private void InvalidateSkeleton() => _boneTree.Invalidate();

    public override void OnInspectorGUI()
    {
        var config = (LmmConfig)target;

        serializedObject.Update();

        // Scoped to the fields: GUI.changed is also set by a button press, so a blanket check would
        // wipe hasTrained the instant Train set it.
        EditorGUI.BeginChangeCheck();
        DrawPropertiesExcluding(serializedObject, "m_Script");
        var fieldsChanged = EditorGUI.EndChangeCheck();

        serializedObject.ApplyModifiedProperties();

        if (fieldsChanged)
        {
            // Which field moved is not knowable here, so every edit invalidates the checkpoint.
            MarkStale(config);
            InvalidateSkeleton();
        }

        if (!config.TryValidate(out var error))
        {
            EditorGUILayout.HelpBox(error, MessageType.Error);
        }

        EditorGUILayout.Space();
        DrawDatabaseSection(config);

        EditorGUILayout.Space();
        DrawBoneSection(config);

        EditorGUILayout.Space();
        DrawTrainingSection(config);

        if (GUI.changed) EditorUtility.SetDirty(config);
    }

    private static void MarkStale(LmmConfig config)
    {
        config.hasTrained = false;
        EditorUtility.SetDirty(config);
    }

    // --- Database -------------------------------------------------------------------------------

    /// <summary>
    /// What the config is learning, and whether it has been generated on its
    /// <c>MotionMatchingData</c>.
    /// </summary>
    private void DrawDatabaseSection(LmmConfig config)
    {
        EditorGUILayout.LabelField("Database", EditorStyles.boldLabel);

        if (config.mmData == null) return;

        var databasePath = config.mmData.GetAssetPath();
        EditorGUILayout.LabelField("Source", ProjectRelative(databasePath));

        var poses = Path.Combine(databasePath, config.mmData.name + ".mmpose");
        var features = Path.Combine(databasePath, config.mmData.name + ".mmfeatures");

        if (File.Exists(poses) && File.Exists(features)) return;

        EditorGUILayout.HelpBox(
            $"\"{config.mmData.name}\" has not been generated. Run MoSynth > Database > Regenerate " +
            "Motion Matching Databases, or press Generate on that asset.", MessageType.Warning);
    }

    // --- Predicted bones ------------------------------------------------------------------------

    /// <summary>
    /// Which bones the decompressor predicts, drawn as the skeleton hierarchy with a row per joint.
    /// </summary>
    private void DrawBoneSection(LmmConfig config)
    {
        _showBones = EditorGUILayout.Foldout(_showBones, "Predicted Bones", true);
        if (!_showBones) return;

        if (!_boneTree.TryRead(config.TryGetDatabaseSkeleton))
        {
            EditorGUILayout.HelpBox(
                "Assign a MotionMatchingData first — its skeleton is the bone list the " +
                "decompressor is defined over.", MessageType.Warning);
            return;
        }

        EditorGUILayout.HelpBox(
            "Bones the decompressor reconstructs. Unticked bones are held at their rest rotation " +
            "and cost the model nothing, which is worth doing for joints that barely move.\n" +
            "Unticking a bone unticks its whole subtree: the model predicts rotations, and a " +
            "rotation needs its parent's frame to be applied in.\n" +
            "Changing the selection changes the model's shape, so it needs a retrain.",
            MessageType.None);

        if (_boneTree.Draw(config, config.excludedBones)) MarkStale(config);
    }

    // --- Training -------------------------------------------------------------------------------

    private void DrawTrainingSection(LmmConfig config)
    {
        EditorGUILayout.LabelField("Networks", EditorStyles.boldLabel);

        var checkpointPath = config.GetCheckpointPath();
        var trained = File.Exists(checkpointPath);

        EditorGUILayout.LabelField("Checkpoint",
            trained
                ? $"{ProjectRelative(checkpointPath)}  ({new FileInfo(checkpointPath).Length / 1024} KB)"
                : "not trained yet");

        if (trained)
        {
            if (config.hasTrained)
            {
                EditorGUILayout.LabelField(" ", "Up to date with this config.", EditorStyles.miniLabel);
            }
            else
            {
                EditorGUILayout.HelpBox(
                    "The config changed since the networks were trained. Until they are retrained " +
                    "the stage will refuse the checkpoint rather than run a model shaped for a " +
                    "different config.", MessageType.Error);
            }
        }

        if (EditorApplication.isPlayingOrWillChangePlaymode)
        {
            EditorGUILayout.HelpBox(
                "Training is disabled in play mode. It would hold the Python GIL for as long as it " +
                "runs and stall the LmmStage.", MessageType.Info);
        }

        using (new EditorGUI.DisabledScope(
                   !config.TryValidate(out _) || EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button("Train LMM", GUILayout.Height(30))) LmmTraining.Run(config);

            // The autoencoder is the slow fit, and the later networks only need its baked latents.
            using (new EditorGUI.DisabledScope(!trained))
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Fit Stepper Only")) LmmTraining.RunStepper(config);
                if (GUILayout.Button("Fit Projector Only")) LmmTraining.RunProjector(config);
            }
        }

        EditorGUILayout.LabelField(
            "Fitting the stepper or the projector alone leaves the rest of the checkpoint " +
            "untouched, so neither makes a stale one current.", EditorStyles.wordWrappedMiniLabel);
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
