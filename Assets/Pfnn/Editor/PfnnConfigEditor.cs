using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AnimationTools;
using AnimationTools.Editor;
using UnityEditor;
using UnityEngine;

namespace Pfnn.Editor
{
/// <summary>
/// Inspector for <see cref="PfnnConfig"/>: extract the pose database, choose which bones the
/// network predicts, then train it over them.
/// </summary>
[CustomEditor(typeof(PfnnConfig))]
public class PfnnConfigEditor : UnityEditor.Editor
{
    private bool _showBones = true;

    // Cached because OnInspectorGUI repaints constantly; dropped on enable and on regeneration.
    private Skeleton _skeleton;
    private int[] _jointDepths;
    private bool _skeletonRead;

    private SerializedProperty _contactBonesProperty;

    private void OnEnable()
    {
        InvalidateSkeleton();
        _contactBonesProperty = serializedObject.FindProperty("contactBones");
    }

    private void InvalidateSkeleton()
    {
        _skeleton = null;
        _jointDepths = null;
        _skeletonRead = false;
    }

    public override void OnInspectorGUI()
    {
        var config = (PfnnConfig)target;

        serializedObject.Update();

        // Scoped to the fields: GUI.changed is also set by a button press, so a blanket check would
        // wipe hasTrained the instant Train set it.
        EditorGUI.BeginChangeCheck();
        DrawPropertiesExcluding(serializedObject, "m_Script", "contactBones");
        PoseSetSourceGUI.DrawContactBones(_contactBonesProperty, config.Skeleton);
        var fieldsChanged = EditorGUI.EndChangeCheck();

        serializedObject.ApplyModifiedProperties();

        if (fieldsChanged)
        {
            // Which field moved is not knowable here, so every edit invalidates both artefacts.
            MarkStale(config, database: true);
        }

        PoseSetSourceGUI.DrawSkeletonValidation(config);

        EditorGUILayout.Space();
        DrawDatabaseSection(config);

        EditorGUILayout.Space();
        DrawBoneSection(config);

        EditorGUILayout.Space();
        DrawTrainingSection(config);

        if (GUI.changed) EditorUtility.SetDirty(config);
    }

    /// <summary>
    /// Note that an artefact no longer matches the config. <paramref name="database"/> extends that
    /// to the pose database, which drags the checkpoint with it: regenerating changes the frames
    /// the network was fitted to.
    /// </summary>
    private static void MarkStale(PfnnConfig config, bool database)
    {
        if (database) config.hasPoseDatabase = false;
        config.hasTrained = false;
        EditorUtility.SetDirty(config);
    }

    private void DrawDatabaseSection(PfnnConfig config)
    {
        EditorGUILayout.LabelField("Pose Database", EditorStyles.boldLabel);
        EditorGUILayout.LabelField("Output", ProjectRelative(config.GetAssetPath()));

        if (File.Exists(config.GetPoseDatabasePath()))
        {
            DrawBuildState(config.hasPoseDatabase,
                "The config changed since the pose database was generated. Regenerate it, then " +
                "retrain — the network is fitted to these frames.");
        }

        // The clips are checked when the button is pressed; see PoseSetSourceGUI.DrawSkeletonValidation.
        using (new EditorGUI.DisabledScope(!PoseSetImporter.TryValidateSettings(config, out _)))
        {
            if (GUILayout.Button("Generate Pose Database", GUILayout.Height(24)))
            {
                GeneratePoseDatabase(config);
                InvalidateSkeleton(); // the joint list the bone rows are drawn from
            }
        }
    }

    /// <summary>Say which button needs pressing, or confirm that none does.</summary>
    private static void DrawBuildState(bool current, string staleMessage)
    {
        if (current)
        {
            EditorGUILayout.LabelField(" ", "Up to date with this config.", EditorStyles.miniLabel);
            return;
        }

        EditorGUILayout.HelpBox(staleMessage, MessageType.Error);
    }

    // --- Predicted bones ------------------------------------------------------------------------

    /// <summary>
    /// Which bones the network predicts, drawn as the skeleton hierarchy with a row per joint.
    /// </summary>
    private void DrawBoneSection(PfnnConfig config)
    {
        _showBones = EditorGUILayout.Foldout(_showBones, "Predicted Bones", true);
        if (!_showBones) return;

        if (!TryReadSkeleton(config))
        {
            EditorGUILayout.HelpBox(
                "Assign this config's skeleton first — it is the bone list the network is defined " +
                "over.", MessageType.Warning);
            return;
        }

        EditorGUILayout.HelpBox(
            "Bones the network predicts. Unticked bones are held at their rest rotation and cost " +
            "the model nothing, which is worth doing for joints that barely move — a few thousand " +
            "frames does not support a large model.\n" +
            "Unticking a bone unticks its whole subtree: the network predicts rotations, and a " +
            "rotation needs its parent's frame to be applied in.\n" +
            "Changing the selection changes the model's shape, so it needs a retrain.",
            MessageType.None);

        DrawBoneRows(config);
        DrawBoneFooter(config);
    }

    private void DrawBoneRows(PfnnConfig config)
    {
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            var boneName = _skeleton.GetBone(i).Name;
            var predicted = config.IsPredicted(boneName);

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Space(_jointDepths[i] * 12f);

                var toggled = EditorGUILayout.ToggleLeft(
                    new GUIContent(boneName, $"joint {i}"), predicted,
                    predicted ? EditorStyles.label : EditorStyles.miniLabel);

                if (toggled == predicted) continue;

                Undo.RecordObject(config, "Edit predicted bones");
                SetSubtreePredicted(config, i, toggled);
                // The extraction does not change, only the model's shape.
                MarkStale(config, database: false);
            }
        }
    }

    /// <summary>
    /// Include or exclude a bone with its whole subtree, keeping the selection closed under parent.
    /// Including one also includes its ancestors, which it needs as a frame.
    /// </summary>
    private void SetSubtreePredicted(PfnnConfig config, int boneIndex, bool predicted)
    {
        foreach (var bone in PfnnBoneSelection.Subtree(_skeleton, boneIndex))
        {
            config.SetPredicted(_skeleton.GetBone(bone).Name, predicted);
        }

        if (!predicted) return;

        for (var parent = _skeleton.GetParentIndex(boneIndex);
             parent >= 0;
             parent = _skeleton.GetParentIndex(parent))
        {
            config.SetPredicted(_skeleton.GetBone(parent).Name, true);
        }
    }

    private void DrawBoneFooter(PfnnConfig config)
    {
        var predicted = 0;
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            if (config.IsPredicted(_skeleton.GetBone(i).Name)) predicted++;
        }

        EditorGUILayout.LabelField(
            $"{predicted} of {_skeleton.BoneCount} bones predicted.", EditorStyles.miniLabel);

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Exclude Fingers And Leaves"))
            {
                Undo.RecordObject(config, "Exclude fingers and leaves");
                PfnnDefaultBoneSelection.Apply(config, _skeleton);
                MarkStale(config, database: false);
            }

            using (new EditorGUI.DisabledScope(!config.HasExcludedBones))
            {
                if (GUILayout.Button("Predict All"))
                {
                    Undo.RecordObject(config, "Predict all bones");
                    config.excludedBones.Clear();
                    MarkStale(config, database: false);
                }
            }

            if (GUILayout.Button("Reload Skeleton")) InvalidateSkeleton();
        }

        // Python refuses names the skeleton does not have, so surface them before training does.
        var orphans = config.excludedBones
            .Where(boneName => !_skeleton.TryFindByName(boneName, out _))
            .ToList();
        if (orphans.Count == 0) return;

        EditorGUILayout.HelpBox(
            $"{orphans.Count} excluded bone(s) are not in this skeleton, which training refuses: " +
            string.Join(", ", orphans), MessageType.Warning);

        if (GUILayout.Button("Remove Stale Entries"))
        {
            Undo.RecordObject(config, "Remove stale excluded bones");
            config.excludedBones.RemoveAll(boneName => !_skeleton.TryFindByName(boneName, out _));
            MarkStale(config, database: false);
        }
    }

    /// <summary>Read and cache the config's skeleton, with each joint's depth for indentation.</summary>
    private bool TryReadSkeleton(PfnnConfig config)
    {
        if (_skeletonRead) return _skeleton != null;
        _skeletonRead = true;

        if (!config.TryGetDatabaseSkeleton(out var skeleton) || skeleton.BoneCount == 0) return false;

        _skeleton = skeleton;
        _jointDepths = new int[skeleton.BoneCount];

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            var depth = 0;
            var parent = skeleton.GetParentIndex(i);
            // Bounded by the joint count so a cyclic parent index cannot hang the inspector.
            while (parent >= 0 && parent < skeleton.BoneCount && depth < skeleton.BoneCount)
            {
                depth++;
                parent = skeleton.GetParentIndex(parent);
            }

            _jointDepths[i] = depth;
        }

        return true;
    }

    // --- Training ---------------------------------------------------------------------------------

    private void DrawTrainingSection(PfnnConfig config)
    {
        EditorGUILayout.LabelField("Network", EditorStyles.boldLabel);

        var checkpointPath = config.GetCheckpointPath();
        var databaseExists = File.Exists(config.GetPoseDatabasePath());
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
                "The config changed since the network was trained. Until it is retrained the stage " +
                "will refuse the checkpoint rather than run a model shaped for a different config.");
        }

        if (EditorApplication.isPlayingOrWillChangePlaymode)
        {
            EditorGUILayout.HelpBox(
                "Training is disabled in play mode. It would hold the Python GIL for as long as it " +
                "runs and stall the PfnnStage.", MessageType.Info);
        }

        using (new EditorGUI.DisabledScope(
                   !databaseExists || EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button("Train PFNN", GUILayout.Height(30))) PfnnTraining.Run(config);
        }
    }

    /// <summary>
    /// Extract and serialize the pose database, and mark any trained network stale. Returns false
    /// if it did not get written.
    /// </summary>
    public static bool GeneratePoseDatabase(PfnnConfig config)
    {
        if (!config.TryValidate(out var error))
        {
            Debug.LogError($"[PFNN] '{config.name}': {error}", config);
            return false;
        }

        try
        {
            EditorUtility.DisplayProgressBar("PFNN", "Extracting poses...", 0.3f);
            config.ImportPoseSet();

            EditorUtility.DisplayProgressBar("PFNN", "Writing database...", 0.7f);
            new PoseSerializer().Serialize(config.GetOrImportPoseSet(), config.GetAssetPath(),
                config.name);

            Debug.Log($"[PFNN] Wrote {config.GetOrImportPoseSet().NumberPoses} poses to " +
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
