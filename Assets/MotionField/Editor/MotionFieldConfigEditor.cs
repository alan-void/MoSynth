using System;
using System.IO;
using System.Linq;
using AnimationTools;
using AnimationTools.Editor;
using Python.Runtime;
using UnityEditor;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionField.Editor
{
/// <summary>
/// Inspector for <see cref="MotionFieldConfig"/>: extract the pose database, then train the value
/// function over it.
///
/// Training runs in-process through PythonNET, synchronously on the main thread. It takes roughly
/// 20 s on a GPU for a 7.8k-state database.
/// </summary>
[CustomEditor(typeof(MotionFieldConfig))]
public class MotionFieldConfigEditor : UnityEditor.Editor
{
    private UnityEngine.Object _importSource;
    private bool _showImport;
    private bool _showBoneWeights = true;

    // Skeleton of the generated database, cached because OnInspectorGUI repaints constantly and
    // this comes off disk. Dropped on enable and whenever the database is regenerated.
    private Skeleton _skeleton;
    private int[] _jointDepths;
    private bool _skeletonRead;

    private SerializedProperty _leftContactBoneProperty;
    private SerializedProperty _rightContactBoneProperty;

    private void OnEnable()
    {
        InvalidateSkeleton();
        _leftContactBoneProperty = serializedObject.FindProperty("leftContactBone");
        _rightContactBoneProperty = serializedObject.FindProperty("rightContactBone");
    }

    private void InvalidateSkeleton()
    {
        _skeleton = null;
        _jointDepths = null;
        _skeletonRead = false;
    }

    public override void OnInspectorGUI()
    {
        var config = (MotionFieldConfig)target;
        var rigRoot = GetRigRoot(config);

        serializedObject.Update();

        // Scoped to the fields on purpose. GUI.changed is also set by a button press, so clearing
        // the flags from the blanket check at the bottom of this method would wipe hasTrained the
        // instant Train Motion Field set it.
        EditorGUI.BeginChangeCheck();
        DrawPropertiesExcluding(serializedObject, "m_Script", "leftContactBone", "rightContactBone");
        DrawContactBones(rigRoot);
        var fieldsChanged = EditorGUI.EndChangeCheck();

        serializedObject.ApplyModifiedProperties();

        if (fieldsChanged)
        {
            // Which field moved is not knowable here, so every edit invalidates both artefacts.
            // Over-flagging costs a rebuild; under-flagging costs a character that quietly moves
            // worse than it should.
            MarkStale(config, database: true);
        }

        DrawSkeletonValidation(config);

        EditorGUILayout.Space();
        DrawImportSection(config);

        EditorGUILayout.Space();
        DrawDatabaseSection(config);

        EditorGUILayout.Space();
        DrawBoneWeightSection(config);

        EditorGUILayout.Space();
        DrawTrainingSection(config);

        EditorGUILayout.Space();
        DrawVisualizationSection(config);

        if (GUI.changed)
        {
            EditorUtility.SetDirty(config);
        }
    }

    // --- Build state ----------------------------------------------------------------------------

    /// <summary>
    /// Note that an artifact no longer matches the config. <paramref name="database"/> extends that
    /// to the pose database, which drags the value function with it: extraction renumbers every
    /// state, and the value function is indexed by state.
    /// </summary>
    private static void MarkStale(MotionFieldConfig config, bool database)
    {
        if (database) config.hasPoseDatabase = false;
        config.hasTrained = false;
        EditorUtility.SetDirty(config);
    }

    /// <summary>Rig a bone picker draws its dropdown from: this config's skeleton root.</summary>
    private static Transform GetRigRoot(MotionFieldConfig config)
    {
        return config.Skeleton != null && config.Skeleton.IsSet ? config.Skeleton.Root : null;
    }

    /// <summary>
    /// One HelpBox per problem the skeleton or an animation clip has, mirroring
    /// <c>MotionMatchingDataEditor</c>'s per-clip validation.
    /// </summary>
    private static void DrawSkeletonValidation(MotionFieldConfig config)
    {
        if (config.Skeleton == null || !config.Skeleton.IsSet)
        {
            EditorGUILayout.HelpBox(
                "No skeleton assigned. Assign the rig's identity armature node as the skeleton root.",
                MessageType.Error);
            return;
        }

        var clips = config.animationClips;
        for (var i = 0; i < clips.Count; i++)
        {
            var clip = clips[i];
            if (clip == null)
            {
                EditorGUILayout.HelpBox($"Animation clip {i} is not assigned.", MessageType.Error);
                continue;
            }

            if (clip.Skeleton == null)
            {
                EditorGUILayout.HelpBox($"Clip \"{clip.name}\" has no resolvable skeleton.", MessageType.Error);
                continue;
            }

            if (!clip.TryValidate(out var error))
            {
                EditorGUILayout.HelpBox($"Clip \"{clip.name}\": {error}", MessageType.Error);
                continue;
            }

            if (!config.Skeleton.MatchesFrom(1, clip.Skeleton))
            {
                EditorGUILayout.HelpBox(
                    $"Clip \"{clip.name}\"'s skeleton does not match this config's skeleton from bone 1 (Hips) onward.",
                    MessageType.Error);
            }
        }
    }

    private void DrawContactBones(Transform rigRoot)
    {
        EditorGUILayout.LabelField("Contact Bones", EditorStyles.boldLabel);
        SkeletonBoneDrawer.DrawLayout(
            new GUIContent("Left Contact Bone", "Bone whose velocity drives foot-contact detection; leave unset to pick by name (LeftToe/RightToe)."),
            _leftContactBoneProperty, rigRoot);
        SkeletonBoneDrawer.DrawLayout(
            new GUIContent("Right Contact Bone", "Bone whose velocity drives foot-contact detection; leave unset to pick by name (LeftToe/RightToe)."),
            _rightContactBoneProperty, rigRoot);
    }

    private void DrawImportSection(MotionFieldConfig config)
    {
        _showImport = EditorGUILayout.Foldout(_showImport, "Import Settings From an IPoseSetSource asset", true);
        if (!_showImport) return;

        EditorGUILayout.HelpBox(
            "One-time copy of the contact threshold and contact bones from any other pose-set source " +
            "asset (e.g. a MotionMatchingData). It does not create a link -- later edits to the " +
            "source asset will not follow.",
            MessageType.Info);

        _importSource = EditorGUILayout.ObjectField(
            "Source", _importSource, typeof(UnityEngine.Object), false);

        var source = _importSource as IPoseSetSource;
        if (_importSource != null && source == null)
        {
            EditorGUILayout.HelpBox(
                $"'{_importSource.name}' does not implement IPoseSetSource.", MessageType.Warning);
        }

        using (new EditorGUI.DisabledScope(source == null))
        {
            if (GUILayout.Button("Copy Settings"))
            {
                var rigRoot = GetRigRoot(config);
                Undo.RecordObject(config, "Import MotionField settings");
                config.contactVelocityThreshold = source.ContactVelocityThreshold;
                config.leftContactBone = ResolveContactBone(source.LeftContactBoneName, rigRoot);
                config.rightContactBone = ResolveContactBone(source.RightContactBoneName, rigRoot);
                MarkStale(config, database: true); // rewrites the clips extraction reads
                Debug.Log($"[MotionField] Copied contact threshold and contact bones from '{source.name}'.");
            }
        }
    }

    /// <summary>
    /// A <see cref="SkeletonBone"/> names a Transform, not a bare string, so a name copied from
    /// another asset has to be resolved against this config's own rig immediately rather than
    /// carried across unresolved. Returns an unset bone when the rig has no bone of that name.
    /// </summary>
    private static SkeletonBone ResolveContactBone(string boneName, Transform rigRoot)
    {
        if (string.IsNullOrEmpty(boneName) || rigRoot == null) return new SkeletonBone();

        foreach (var transform in Skeleton.CollectTransformsDfs(rigRoot))
        {
            if (transform.name == boneName) return new SkeletonBone(new Skeleton(rigRoot), transform);
        }

        Debug.LogWarning($"[MotionField] Could not find bone \"{boneName}\" under rig \"{rigRoot.name}\" to copy.");
        return new SkeletonBone();
    }

    private void DrawDatabaseSection(MotionFieldConfig config)
    {
        EditorGUILayout.LabelField("Pose Database", EditorStyles.boldLabel);
        EditorGUILayout.LabelField("Output", ProjectRelative(config.GetAssetPath()));

        bool hasClips = config.animationClips != null && config.animationClips.Count > 0;
        if (!hasClips)
        {
            EditorGUILayout.HelpBox("Assign at least one animation clip.", MessageType.Warning);
        }

        if (File.Exists(config.GetPoseDatabasePath()))
        {
            DrawBuildState(config.hasPoseDatabase,
                "The config changed since the pose database was generated. Regenerate it, then " +
                "retrain -- extraction renumbers every state, which invalidates the value function " +
                "too.");
        }

        // Gated on the same check ImportPoseSet runs, so a config with no skeleton -- or one its
        // clips do not line up with -- cannot be generated into a database that would fail to load.
        // DrawSkeletonValidation has already put the reason on screen. Matches
        // MotionMatchingDataEditor, whose button is disabled by _generateButtonError.
        using (new EditorGUI.DisabledScope(!config.TryValidate(out _)))
        {
            if (GUILayout.Button("Generate Pose Database", GUILayout.Height(24)))
            {
                if (GeneratePoseDatabase(config))
                {
                    config.hasPoseDatabase = true;
                    config.hasTrained = false; // the states the value function indexes were renumbered
                    EditorUtility.SetDirty(config);
                    AssetDatabase.SaveAssetIfDirty(config);
                }

                InvalidateSkeleton(); // the joint list the bone weight rows are drawn from
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

    /// <summary>
    /// Per-joint weights, drawn as the skeleton hierarchy.
    /// </summary>
    /// <remarks>
    /// The list on the asset is name-keyed and sparse, so a default list drawer would show a
    /// handful of anonymous elements in whatever order they were edited. Here every joint of the
    /// database gets a row, indented by its depth, whether or not it carries a stored weight.
    /// </remarks>
    private void DrawBoneWeightSection(MotionFieldConfig config)
    {
        _showBoneWeights = EditorGUILayout.Foldout(
            _showBoneWeights, "Bone Weights (similarity metric)", true);
        if (!_showBoneWeights) return;

        if (!TryReadSkeleton(config))
        {
            EditorGUILayout.HelpBox(
                "Assign this config's skeleton first — it is the bone list the similarity metric " +
                "is computed over.",
                MessageType.Warning);
            return;
        }

        EditorGUILayout.HelpBox(
            "Multiplies each joint's contribution to the k-NN distance, on top of the global " +
            "Pos Weight and Vel Weight. The metric sums over joints, so lowering one stops it " +
            "outvoting the joints you care about; 0 removes it from matching entirely.\n" +
            "This changes the metric, so retrain the value function afterwards -- until then " +
            "the stage rejects the stale one and falls back to greedy control.",
            MessageType.None);

        DrawBoneWeightRows(config);
        DrawBoneWeightFooter(config);
    }

    private void DrawBoneWeightRows(MotionFieldConfig config)
    {
        const float fieldWidth = 54f;

        using (new EditorGUILayout.HorizontalScope())
        {
            EditorGUILayout.LabelField("Joint", EditorStyles.miniBoldLabel);
            EditorGUILayout.LabelField("Pos", EditorStyles.miniBoldLabel, GUILayout.Width(fieldWidth));
            EditorGUILayout.LabelField("Vel", EditorStyles.miniBoldLabel, GUILayout.Width(fieldWidth));
        }

        for (int i = 0; i < _skeleton.BoneCount; i++)
        {
            var bone = _skeleton.GetBone(i);
            MotionFieldConfig.BoneWeight weight = config.GetBoneWeight(bone.Name);

            // Row 0 is the simulation bone. Forward kinematics pins it to the origin with identity
            // rotation before the feature is built, so its position and velocity are structurally
            // zero and no weight can change them. Shown for orientation, not editable.
            bool isRoot = i == 0;

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Space(_jointDepths[i] * 12f);

                var label = new GUIContent(bone.Name, isRoot
                    ? "The root is pinned to the origin when the metric is built, so its rows are " +
                      "always zero and its weight has no effect."
                    : $"joint {i}");
                EditorGUILayout.LabelField(label,
                    weight.IsNeutral ? EditorStyles.label : EditorStyles.boldLabel);

                using (new EditorGUI.DisabledScope(isRoot))
                {
                    float position = EditorGUILayout.FloatField(
                        weight.position, GUILayout.Width(fieldWidth));
                    float velocity = EditorGUILayout.FloatField(
                        weight.velocity, GUILayout.Width(fieldWidth));

                    if (isRoot) continue;
                    if (Mathf.Approximately(position, weight.position) &&
                        Mathf.Approximately(velocity, weight.velocity)) continue;

                    Undo.RecordObject(config, "Edit bone weight");
                    config.SetBoneWeight(new MotionFieldConfig.BoneWeight(
                        bone.Name, Mathf.Max(0f, position), Mathf.Max(0f, velocity)));
                    // The metric changes, the extraction does not, so only the training is stale.
                    MarkStale(config, database: false);
                }
            }
        }
    }

    private void DrawBoneWeightFooter(MotionFieldConfig config)
    {
        string[] weighted = config.boneWeights
            .Where(w => !w.IsNeutral)
            .Select(w => $"{w.name} ({w.position:0.##}/{w.velocity:0.##})")
            .ToArray();

        EditorGUILayout.LabelField(
            weighted.Length == 0 ? "All joints at 1 -- metric unchanged." : string.Join(", ", weighted),
            EditorStyles.wordWrappedMiniLabel);

        using (new EditorGUILayout.HorizontalScope())
        {
            using (new EditorGUI.DisabledScope(weighted.Length == 0))
            {
                if (GUILayout.Button("Reset All To 1"))
                {
                    Undo.RecordObject(config, "Reset bone weights");
                    config.boneWeights.Clear();
                    MarkStale(config, database: false);
                }
            }

            if (GUILayout.Button("Reload Skeleton"))
            {
                InvalidateSkeleton();
            }
        }

        // Entries naming joints the current skeleton does not have. They are ignored at runtime
        // (Python logs them), but silently keeping them would make the summary above lie.
        var orphans = config.boneWeights
            .Where(w => !_skeleton.TryFindByName(w.name, out _))
            .ToList();
        if (orphans.Count == 0) return;

        EditorGUILayout.HelpBox(
            $"{orphans.Count} weight(s) name joints this skeleton does not have and are ignored: " +
            string.Join(", ", orphans.Select(w => w.name)), MessageType.Warning);

        if (GUILayout.Button("Remove Stale Entries"))
        {
            Undo.RecordObject(config, "Remove stale bone weights");
            // Training ignores these already, so this cannot change a result -- but the flag is
            // cheap and a config that says "retrain" when nothing needs it is the harmless way
            // to be wrong.
            config.boneWeights.RemoveAll(w => !_skeleton.TryFindByName(w.name, out _));
            MarkStale(config, database: false);
        }
    }

    /// <summary>Read and cache the database skeleton, with each joint's depth for indentation.</summary>
    private bool TryReadSkeleton(MotionFieldConfig config)
    {
        if (_skeletonRead) return _skeleton != null;
        _skeletonRead = true;

        if (!config.TryGetDatabaseSkeleton(out Skeleton skeleton) || skeleton.BoneCount == 0)
        {
            return false;
        }

        _skeleton = skeleton;
        _jointDepths = new int[skeleton.BoneCount];

        for (int i = 0; i < skeleton.BoneCount; i++)
        {
            int depth = 0;
            int parent = skeleton.GetParentIndex(i);
            // Bounded by the joint count so a cyclic parentIndex cannot hang the inspector.
            while (parent >= 0 && parent < skeleton.BoneCount && depth < skeleton.BoneCount)
            {
                depth++;
                parent = skeleton.GetParentIndex(parent);
            }

            _jointDepths[i] = depth;
        }

        return true;
    }

    private void DrawTrainingSection(MotionFieldConfig config)
    {
        EditorGUILayout.LabelField("Value Function", EditorStyles.boldLabel);

        string valuePath = config.GetValueFunctionPath();
        bool databaseExists = File.Exists(config.GetPoseDatabasePath());
        bool trained = File.Exists(valuePath);

        EditorGUILayout.LabelField("Trained file",
            trained
                ? $"{ProjectRelative(valuePath)}  ({new FileInfo(valuePath).Length / 1024} KB)"
                : "not trained yet");

        if (!databaseExists)
        {
            EditorGUILayout.HelpBox("Generate the pose database first.", MessageType.Warning);
        }
        else if (trained)
        {
            DrawBuildState(config.hasTrained,
                "The config changed since the value function was trained. Until it is retrained " +
                "the stage rejects it and runs the one-step-greedy policy.");
        }

        if (EditorApplication.isPlayingOrWillChangePlaymode)
        {
            EditorGUILayout.HelpBox(
                "Training is disabled in play mode. It would hold the Python GIL for seconds at a " +
                "time and stall the running MotionFieldStage.", MessageType.Info);
        }

        using (new EditorGUI.DisabledScope(!databaseExists || EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button("Train Motion Field", GUILayout.Height(30)))
            {
                TrainMotionField(config);
            }
        }
    }

    private void DrawVisualizationSection(MotionFieldConfig config)
    {
        EditorGUILayout.LabelField("Debug Visualization", EditorStyles.boldLabel);

        string embeddingPath = config.GetEmbeddingPath();
        bool databaseExists = File.Exists(config.GetPoseDatabasePath());
        bool embedded = File.Exists(embeddingPath);

        EditorGUILayout.LabelField("Embedding",
            embedded
                ? $"{ProjectRelative(embeddingPath)}  ({new FileInfo(embeddingPath).Length / 1024} KB)"
                : "not computed yet");

        if (!databaseExists)
        {
            EditorGUILayout.HelpBox("Generate the pose database first.", MessageType.Warning);
        }

        EditorGUILayout.HelpBox(
            "Projects the motion field to 3D so MotionFieldVisualizer can draw it. Independent of " +
            "training -- the value function does not need it, and deleting it only turns the " +
            "visualizer off. Expect ~20 s: the first UMAP import pays a one-off numba warm-up.\n\n" +
            "Recompute after changing any of the settings above. An embedding written by an older " +
            "version of this tool is rejected outright rather than reused, so the visualizer will " +
            "draw nothing until it is rebuilt.",
            MessageType.None);

        using (new EditorGUI.DisabledScope(!databaseExists || EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button("Compute UMAP Embedding", GUILayout.Height(30)))
            {
                ComputeEmbedding(config);
            }
        }
    }

    /// <summary>Extract and serialize the pose database. Returns false if it did not get written.</summary>
    private static bool GeneratePoseDatabase(MotionFieldConfig config)
    {
        try
        {
            EditorUtility.DisplayProgressBar("Motion Field", "Extracting poses...", 0.3f);
            config.ImportPoseSet();

            EditorUtility.DisplayProgressBar("Motion Field", "Writing database...", 0.7f);
            new PoseSerializer().Serialize(config.GetOrImportPoseSet(), config.GetAssetPath(), config.name);

            Debug.Log($"[MotionField] Wrote {config.GetOrImportPoseSet().NumberPoses} poses to " +
                      $"{ProjectRelative(config.GetAssetPath())}.");
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

    private void TrainMotionField(MotionFieldConfig config)
    {
        // The interpreter runs in this process. A domain reload while it is mid-call takes the
        // editor down with it, so hold reloads off for the duration.
        EditorApplication.LockReloadAssemblies();
        try
        {
            PythonRuntime.EnsureInitialized(config.pythonDllPath, config.pythonVenvPath);

            // Marshalled into Python as a callable. Training is synchronous on the main thread, so
            // this fires on the main thread too and may touch the editor UI directly.
            Action<string, double> report = (stage, fraction) =>
                EditorUtility.DisplayProgressBar("Training Motion Field", stage, (float)fraction);

            using (Py.GIL())
            {
                PyObject trainer = PythonRuntime.Import("motion_field_trainer", reload: true);

                using var args = new PyTuple(new[]
                {
                    config.GetAssetPath().ToPython(),
                    config.name.ToPython(),
                    config.GetValueFunctionPath().ToPython(),
                });

                using var kwargs = new PyDict();
                kwargs["k_neighbors"] = config.kNeighbors.ToPython();
                kwargs["theta_count"] = config.thetaCount.ToPython();
                kwargs["epochs"] = config.epochs.ToPython();
                kwargs["gamma"] = ((double)config.gamma).ToPython();
                kwargs["tug_ratio"] = ((double)config.tugRatio).ToPython();
                kwargs["pos_weight"] = ((double)config.posWeight).ToPython();
                kwargs["vel_weight"] = ((double)config.velWeight).ToPython();
                kwargs["bone_weights"] = MotionFieldBoneWeights.ToPython(config);
                kwargs["locomotion_factor"] = ((double)config.locomotionFactor).ToPython();
                kwargs["locomotion_speed_threshold"] =
                    ((double)config.locomotionSpeedThreshold).ToPython();
                kwargs["device"] = config.DeviceName.ToPython();
                kwargs["knn_chunk"] = config.knnChunk.ToPython();
                kwargs["state_chunk"] = config.stateChunk.ToPython();
                kwargs["progress"] = report.ToPython();

                using PyObject summary = trainer.InvokeMethod("train", args, kwargs);
                Debug.Log($"[MotionField] {summary}");

                // Only on the success path: train() throwing leaves the old value function on disk,
                // and it is no less stale than it was a moment ago.
                config.hasTrained = true;
                EditorUtility.SetDirty(config);
                AssetDatabase.SaveAssetIfDirty(config);
            }

            GC.KeepAlive(report);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            EditorApplication.UnlockReloadAssemblies();
            AssetDatabase.Refresh();
        }
    }

    /// <summary>
    /// Fit the UMAP projection of the database. Same execution model as
    /// <see cref="TrainMotionField"/>: in-process, synchronous, reloads locked out.
    /// </summary>
    private static void ComputeEmbedding(MotionFieldConfig config)
    {
        EditorApplication.LockReloadAssemblies();
        try
        {
            PythonRuntime.EnsureInitialized(config.pythonDllPath, config.pythonVenvPath);

            Action<string, double> report = (stage, fraction) =>
                EditorUtility.DisplayProgressBar("Embedding Motion Field", stage, (float)fraction);

            using (Py.GIL())
            {
                PyObject embedding = PythonRuntime.Import("motion_field_embedding", reload: true);

                using var args = new PyTuple(new[]
                {
                    config.GetAssetPath().ToPython(),
                    config.name.ToPython(),
                    config.GetEmbeddingPath().ToPython(),
                });

                using var kwargs = new PyDict();
                kwargs["feature_mode"] = config.UmapFeatureModeName.ToPython();
                kwargs["n_components"] = config.umapComponents.ToPython();
                kwargs["n_neighbors"] = config.umapNeighbors.ToPython();
                kwargs["min_dist"] = ((double)config.umapMinDist).ToPython();
                kwargs["device"] = config.DeviceName.ToPython();
                kwargs["seed"] = config.umapSeed.ToPython();
                kwargs["knn_chunk"] = config.knnChunk.ToPython();
                kwargs["pos_weight"] = ((double)config.posWeight).ToPython();
                kwargs["vel_weight"] = ((double)config.velWeight).ToPython();
                kwargs["bone_weights"] = MotionFieldBoneWeights.ToPython(config);
                kwargs["progress"] = report.ToPython();

                using PyObject summary = embedding.InvokeMethod("compute_embedding", args, kwargs);
                Debug.Log($"[MotionField] {summary}");
            }

            GC.KeepAlive(report);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            EditorApplication.UnlockReloadAssemblies();
            AssetDatabase.Refresh();
        }
    }

    private static string ProjectRelative(string absolute)
    {
        string root = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        string full = Path.GetFullPath(absolute);
        return full.StartsWith(root, StringComparison.OrdinalIgnoreCase)
            ? full.Substring(root.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            : full;
    }
}
}