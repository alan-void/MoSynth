using System;
using System.IO;
using System.Linq;
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

    // Skeleton of the assigned database, cached because OnInspectorGUI repaints constantly.
    private Skeleton _skeleton;
    private int[] _jointDepths;
    private bool _skeletonRead;

    private void OnEnable() => InvalidateSkeleton();

    private void InvalidateSkeleton()
    {
        _skeleton = null;
        _jointDepths = null;
        _skeletonRead = false;
    }

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

        if (!TryReadSkeleton(config))
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

        DrawBoneRows(config);
        DrawBoneFooter(config);
    }

    private void DrawBoneRows(LmmConfig config)
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
                MarkStale(config);
            }
        }
    }

    /// <summary>
    /// Include or exclude a bone with its whole subtree, keeping the selection closed under parent.
    /// Including one also includes its ancestors, which it needs as a frame.
    /// </summary>
    private void SetSubtreePredicted(LmmConfig config, int boneIndex, bool predicted)
    {
        foreach (var bone in PredictedBoneSelection.Subtree(_skeleton, boneIndex))
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

    private void DrawBoneFooter(LmmConfig config)
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
                config.excludedBones.Clear();
                foreach (var boneName in DefaultPredictedBones.Excluded(_skeleton))
                {
                    config.SetPredicted(boneName, false);
                }

                MarkStale(config);
            }

            using (new EditorGUI.DisabledScope(!config.HasExcludedBones))
            {
                if (GUILayout.Button("Predict All"))
                {
                    Undo.RecordObject(config, "Predict all bones");
                    config.excludedBones.Clear();
                    MarkStale(config);
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
            MarkStale(config);
        }
    }

    /// <summary>Read and cache the database's skeleton, with each joint's depth for indentation.</summary>
    private bool TryReadSkeleton(LmmConfig config)
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
