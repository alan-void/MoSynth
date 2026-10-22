using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using AnimationTools;
using AnimationTools.Editor;
using GameplayTags;
using GameplayTags.Editor;
using MotionMatching;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Style.Editor
{
/// <summary>
/// Takes the retargeted 100STYLE FBX on Y Bot to a trained <see cref="StyleTransferConfig"/> and a
/// motion matching character that runs it, one idempotent step per entry point:
/// <c>-executeMethod Style.Editor.HundredStylePipeline.CreateClips</c>, <c>.SetUpClips</c>,
/// <c>.CreateConfig</c>, <c>.Bake</c>, <c>.Train</c>, <c>.CreateMmCharacter</c>, or <c>.RunAll</c>.
/// </summary>
/// <remarks>
/// Every step is synchronous, because a batchmode Editor does not pump
/// <see cref="EditorApplication.delayCall"/>, and in batchmode exits with 0 on success and 1
/// otherwise. The Y Bot clips are imported uncompressed, so the clip steps touch one FBX or style
/// folder at a time and unload between them.
/// </remarks>
public static class HundredStylePipeline
{
    private const string LogPrefix = "[Style Pipeline]";
    private const string MenuRoot = "MoSynth/Style/Pipeline/";

    private const string FbxFolder = "Assets/LFS/Animation/YBot/100style";
    private const string FbxPrefix = "100style_";
    private const string RestPoseTake = "T-Pose";
    private const string ClipsRoot = "Assets/Animation/YBot/100style";

    private const string ConfigFolder = "Assets/Animation/Style";
    private const string ConfigPath = ConfigFolder + "/YBot100Style.asset";

    private const string BasePrefabPath = "Assets/Prefabs/YBot/Mm_YBot.prefab";
    private const string VariantPath = "Assets/Prefabs/YBot/Mm_YBot_Style.prefab";
    private const string ScenePath = "Assets/Scenes/StyleTransferMM.unity";

    private const string NeutralStyleName = "Neutral";
    private const string DemoStyleName = "Old";

    /// <summary>Styles defined by cadence or support pattern, which the stage cannot carry.</summary>
    private static readonly string[] ExcludedStyleNames =
        { "LeftHop", "RightHop", "Skip", "TwoFootJump", "SpinAntiClock", "SpinClock", "StartStop" };

    [MenuItem(MenuRoot + "1 Create Clips")]
    public static void CreateClips() => RunStep("Create Clips", CreateClipsStep);

    [MenuItem(MenuRoot + "2 Set Up Clips")]
    public static void SetUpClips() => RunStep("Set Up Clips", SetUpClipsStep);

    [MenuItem(MenuRoot + "3 Create Config")]
    public static void CreateConfig() => RunStep("Create Config", CreateConfigStep);

    [MenuItem(MenuRoot + "4 Bake")]
    public static void Bake() => RunStep("Bake", BakeStep);

    [MenuItem(MenuRoot + "5 Train")]
    public static void Train() => RunStep("Train", TrainStep);

    [MenuItem(MenuRoot + "6 Create MM Character")]
    public static void CreateMmCharacter() => RunStep("Create MM Character", CreateMmCharacterStep);

    [MenuItem(MenuRoot + "Run All")]
    public static void RunAll() => RunStep("Run All", RunAllSteps);

    private static void RunStep(string stepName, Func<bool> step)
    {
        var ok = false;
        var watch = Stopwatch.StartNew();
        try
        {
            ok = step();
        }
        catch (Exception e)
        {
            Debug.LogException(e);
        }

        var minutes = watch.Elapsed.TotalMinutes;
        if (ok) Debug.Log($"{LogPrefix} {stepName} succeeded ({minutes:0.0} min).");
        else Debug.LogError($"{LogPrefix} {stepName} failed ({minutes:0.0} min; see the errors above).");

        if (Application.isBatchMode) EditorApplication.Exit(ok ? 0 : 1);
    }

    private static bool RunAllSteps()
    {
        var steps = new (string name, Func<bool> run)[]
        {
            ("Create Clips", CreateClipsStep),
            ("Set Up Clips", SetUpClipsStep),
            ("Create Config", CreateConfigStep),
            ("Bake", BakeStep),
            ("Train", TrainStep),
            ("Create MM Character", CreateMmCharacterStep),
        };

        foreach (var (name, run) in steps)
        {
            Debug.Log($"{LogPrefix} BEGIN {name}");
            if (run()) continue;

            Debug.LogError($"{LogPrefix} {name} failed; the remaining steps were not run.");
            return false;
        }

        return true;
    }

    // --- 1. Create clips ----------------------------------------------------------------------------

    private static bool CreateClipsStep()
    {
        // The FBX may have been written by a retarget batch since the Editor last looked.
        AssetDatabase.Refresh();

        var fbxPaths = FindStyleFbx();
        if (fbxPaths.Count == 0)
        {
            Debug.LogError($"{LogPrefix} No {FbxPrefix}<Style>.fbx in {FbxFolder}.");
            return false;
        }

        int created = 0, updated = 0, skipped = 0;
        var failed = new List<string>();
        for (var i = 0; i < fbxPaths.Count; i++)
        {
            var fbxPath = fbxPaths[i];
            EditorUtility.DisplayProgressBar("Create 100STYLE Clips", Path.GetFileName(fbxPath),
                (float)i / fbxPaths.Count);
            try
            {
                switch (CreateClipsFor(fbxPath, out var fileCreated, out var fileUpdated, out var error))
                {
                    case FileOutcome.Skipped:
                        skipped++;
                        break;
                    case FileOutcome.Done:
                        created += fileCreated;
                        updated += fileUpdated;
                        Debug.Log($"{LogPrefix} {Path.GetFileName(fbxPath)}: {fileCreated} created, " +
                                  $"{fileUpdated} updated.");
                        break;
                    default:
                        failed.Add(Path.GetFileName(fbxPath));
                        Debug.LogError($"{LogPrefix} {Path.GetFileName(fbxPath)}: {error}");
                        break;
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                ReleaseUnusedAssets();
            }
        }

        Debug.Log($"{LogPrefix} Create Clips: {fbxPaths.Count} FBX, {skipped} already complete, " +
                  $"{created} clips created, {updated} updated, {failed.Count} failed" +
                  (failed.Count > 0 ? $" ({string.Join(", ", failed)})." : "."));
        return failed.Count == 0;
    }

    private enum FileOutcome
    {
        Done,
        Skipped,
        Failed
    }

    /// <summary>The style FBX, sorted, as project-relative paths.</summary>
    private static List<string> FindStyleFbx()
    {
        var absoluteFolder = Path.Combine(ProjectRoot, FbxFolder);
        if (!Directory.Exists(absoluteFolder)) return new List<string>();

        return Directory.GetFiles(absoluteFolder, FbxPrefix + "*.fbx")
            .Select(path => $"{FbxFolder}/{Path.GetFileName(path)}")
            .Where(path => IsSelected(StyleOfFbx(path)))
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToList();
    }

    private const string StylesArgument = "-styles";

    /// <summary>
    /// The styles named by a <c>-styles A,B,C</c> command-line argument, or null for every style.
    /// Lets a batchmode run work on a subset while the rest of the dataset is still retargeting.
    /// </summary>
    private static HashSet<string> SelectedStyles()
    {
        var args = Environment.GetCommandLineArgs();
        var at = Array.IndexOf(args, StylesArgument);
        if (at < 0 || at + 1 >= args.Length) return null;

        return new HashSet<string>(
            args[at + 1].Split(',').Select(style => style.Trim()).Where(style => style.Length > 0),
            StringComparer.OrdinalIgnoreCase);
    }

    private static bool IsSelected(string style)
    {
        var selected = SelectedStyles();
        return selected == null || selected.Contains(style);
    }

    private static string StyleOfFbx(string fbxPath) =>
        Path.GetFileNameWithoutExtension(fbxPath).Substring(FbxPrefix.Length);

    private static string ClipFolderOf(string style) => $"{ClipsRoot}/{style}";

    /// <summary>
    /// One annotated clip per take but the rest pose, updated in place where it exists. A file
    /// whose every take already has its asset is skipped without loading its clips.
    /// </summary>
    private static FileOutcome CreateClipsFor(string fbxPath, out int created, out int updated, out string error)
    {
        created = 0;
        updated = 0;
        error = null;

        if (AssetImporter.GetAtPath(fbxPath) is not ModelImporter importer)
        {
            error = "not imported as a model.";
            return FileOutcome.Failed;
        }

        // Matches the rest of the Y Bot library; keyframe reduction loses fidelity in the bake.
        if (importer.animationCompression != ModelImporterAnimationCompression.Off)
        {
            importer.animationCompression = ModelImporterAnimationCompression.Off;
            importer.SaveAndReimport();
            Debug.Log($"{LogPrefix} {Path.GetFileName(fbxPath)}: set Anim. Compression to Off and reimported.");
        }

        var folder = ClipFolderOf(StyleOfFbx(fbxPath));
        var takes = importer.clipAnimations.Length > 0 ? importer.clipAnimations : importer.defaultClipAnimations;
        var takeNames = takes.Select(t => t.name).Where(IsStyledTake).ToList();
        if (takeNames.Count > 0 && takeNames.All(name => AssetExists(AnnotatedClipFactory.AssetPathFor(folder, name))))
        {
            return FileOutcome.Skipped;
        }

        var model = AssetDatabase.LoadAssetAtPath<GameObject>(fbxPath);
        if (!AnnotatedClipFactory.TryResolveModel(model, out var rootBone, out var clips, out error))
        {
            return FileOutcome.Failed;
        }

        if (!AnnotatedClipFactory.EnsureFolder(folder))
        {
            error = $"could not create the folder '{folder}'.";
            return FileOutcome.Failed;
        }

        // No gait phase here: the frame cuts that Set Up Clips applies decide what detection reads.
        AssetDatabase.StartAssetEditing();
        try
        {
            foreach (var clip in clips)
            {
                if (!IsStyledTake(clip.name)) continue;

                AnnotatedClipFactory.CreateOrUpdate(clip, rootBone,
                    AnnotatedClipFactory.AssetPathFor(folder, clip.name), out var wasCreated);
                if (wasCreated) created++;
                else updated++;
            }
        }
        finally
        {
            AssetDatabase.StopAssetEditing();
            AssetDatabase.SaveAssets();
        }

        if (created + updated == 0)
        {
            error = $"no take other than '{RestPoseTake}'.";
            return FileOutcome.Failed;
        }

        return FileOutcome.Done;
    }

    private static bool IsStyledTake(string takeName) =>
        !string.Equals(takeName, RestPoseTake, StringComparison.OrdinalIgnoreCase);

    // --- 2. Set up clips ----------------------------------------------------------------------------

    private static bool SetUpClipsStep()
    {
        if (!HundredStyleClipSetup.TryLoadDataset(out var dataset, out var error))
        {
            Debug.LogError($"{LogPrefix} {error}");
            return false;
        }

        var folders = StyleFolders();
        if (folders.Count == 0)
        {
            Debug.LogError($"{LogPrefix} No style folders under {ClipsRoot}; run Create Clips first.");
            return false;
        }

        var report = new HundredStyleClipSetup.Report();
        foreach (var folder in folders)
        {
            try
            {
                var clips = LoadClips(folder);
                HundredStyleClipSetup.SetUp(clips, dataset, report);
                Debug.Log($"{LogPrefix} {folder}: {clips.Count} clips set up.");
            }
            finally
            {
                // Each clip was snapshotted for undo, which would otherwise pile up across every folder.
                Undo.ClearAll();
                ReleaseUnusedAssets();
            }
        }

        Debug.Log($"{LogPrefix} Set Up Clips over {folders.Count} style folders: {report.Summary()}");
        if (report.FootfallsFailed > 0)
        {
            Debug.LogWarning($"{LogPrefix} Footfall detection failed on {report.FootfallsFailed} clips; " +
                             "they bake with no anchors. Fix them in the clip editor if they matter.");
        }

        if (report.Unparsed.Count > 0)
        {
            Debug.LogError($"{LogPrefix} Clip names not of the form <Style>_<Gait>, or naming a style " +
                           $"the dataset does not list: {string.Join(", ", report.Unparsed)}");
            return false;
        }

        return report.Clips > 0;
    }

    /// <summary>Every selected style folder under the clips root, sorted by style name.</summary>
    private static List<string> StyleFolders()
    {
        if (!AssetDatabase.IsValidFolder(ClipsRoot)) return new List<string>();

        return AssetDatabase.GetSubFolders(ClipsRoot)
            .Where(folder => IsSelected(Path.GetFileName(folder)))
            .OrderBy(folder => folder, StringComparer.Ordinal)
            .ToList();
    }

    /// <summary>The annotated clips in a folder, sorted by asset path.</summary>
    private static List<AnnotatedAnimationClip> LoadClips(string folder) =>
        AssetDatabase.FindAssets($"t:{nameof(AnnotatedAnimationClip)}", new[] { folder })
            .Select(guid => AssetDatabase.GUIDToAssetPath(guid))
            .OrderBy(path => path, StringComparer.Ordinal)
            .Select(path => AssetDatabase.LoadAssetAtPath<AnnotatedAnimationClip>(path))
            .Where(clip => clip != null)
            .ToList();

    // --- 3. Create config ---------------------------------------------------------------------------

    private static bool CreateConfigStep()
    {
        var mmData = FindBaseMotionMatchingData();
        if (mmData == null) return false;

        var tagConfig = GameplayTagConfig.instance;
        var styleRoot = tagConfig.GetTag(HundredStyleClipSetup.StyleRoot);
        var gaitRoot = tagConfig.GetTag(HundredStyleClipSetup.GaitRoot);
        var neutralStyle = tagConfig.GetTag(StyleTagName(NeutralStyleName));
        if (styleRoot == null || gaitRoot == null || neutralStyle == null)
        {
            Debug.LogError($"{LogPrefix} Tags '{HundredStyleClipSetup.StyleRoot}', " +
                           $"'{HundredStyleClipSetup.GaitRoot}' and '{StyleTagName(NeutralStyleName)}' " +
                           "are needed; run Set Up Clips first.");
            return false;
        }

        // Style folders are sorted, and so are the clips in each, whose names are <Style>_<Gait>.
        // Excluded styles are left out of the database too, since every clip costs bake memory.
        var clips = new List<AnnotatedAnimationClip>();
        foreach (var folder in StyleFolders())
        {
            if (ExcludedStyleNames.Contains(Path.GetFileName(folder))) continue;
            clips.AddRange(LoadClips(folder));
        }
        if (clips.Count == 0)
        {
            Debug.LogError($"{LogPrefix} No annotated clips under {ClipsRoot}; run Create Clips first.");
            return false;
        }

        if (!AnnotatedClipFactory.EnsureFolder(ConfigFolder))
        {
            Debug.LogError($"{LogPrefix} Could not create the folder '{ConfigFolder}'.");
            return false;
        }

        var config = AssetDatabase.LoadAssetAtPath<StyleTransferConfig>(ConfigPath);
        var created = config == null;
        if (created)
        {
            config = ScriptableObject.CreateInstance<StyleTransferConfig>();
            AssetDatabase.CreateAsset(config, ConfigPath);
        }

        var clipsChanged = !config.animationClips.SequenceEqual(clips);

        // Both fields are copied as serialized data, so the config shares no instance with the MM asset.
        using (var source = new SerializedObject(mmData))
        using (var target = new SerializedObject(config))
        {
            target.CopyFromSerializedProperty(source.FindProperty("skeleton"));
            target.CopyFromSerializedProperty(source.FindProperty("contactBones"));
            target.ApplyModifiedPropertiesWithoutUndo();
        }

        config.animationClips = clips;
        config.mirrorClips = false;
        config.contactVelocityThreshold = HundredStyleClipSetup.ContactVelocityThreshold;
        config.styleRoot = styleRoot;
        config.gaitRoot = gaitRoot;
        config.neutralStyle = neutralStyle;
        config.excludedStyles = FindExcludedStyles(tagConfig);
        config.excludedBones = DefaultPredictedBones.Excluded(config.Skeleton);

        if (created || clipsChanged)
        {
            config.hasPoseDatabase = false;
            config.hasTrained = false;
        }

        EditorUtility.SetDirty(config);
        AssetDatabase.SaveAssetIfDirty(config);

        if (!config.TryValidate(out var error))
        {
            Debug.LogError($"{LogPrefix} {ConfigPath} does not validate: {error}", config);
            return false;
        }

        Debug.Log($"{LogPrefix} {(created ? "Created" : "Updated")} {ConfigPath}: {clips.Count} clips over " +
                  $"{StyleFolders().Count} styles, skeleton and {config.contactBones.Count} contact bones from " +
                  $"'{mmData.name}', {config.excludedStyles.Count} styles and {config.excludedBones.Count} " +
                  "bones excluded.");
        return true;
    }

    private static string StyleTagName(string style) => $"{HundredStyleClipSetup.StyleRoot}.{style}";

    private static List<GameplayTagSO> FindExcludedStyles(GameplayTagConfig tagConfig)
    {
        var tags = new List<GameplayTagSO>();
        foreach (var style in ExcludedStyleNames)
        {
            var tag = tagConfig.GetTag(StyleTagName(style));
            if (tag != null) tags.Add(tag);
            else Debug.LogWarning($"{LogPrefix} No tag '{StyleTagName(style)}' to exclude; is its FBX missing?");
        }

        return tags;
    }

    /// <summary>The database the base MM character searches, whose skeleton and contacts the config shares.</summary>
    private static MotionMatchingData FindBaseMotionMatchingData()
    {
        var basePrefab = AssetDatabase.LoadAssetAtPath<GameObject>(BasePrefabPath);
        if (basePrefab == null)
        {
            Debug.LogError($"{LogPrefix} No prefab at {BasePrefabPath}.");
            return null;
        }

        foreach (var synthesizer in basePrefab.GetComponentsInChildren<MotionSynthesisComponent>(true))
        {
            foreach (var stage in synthesizer.stages)
            {
                if (stage is MotionMatchingStage motionMatching && motionMatching.mmData != null)
                    return motionMatching.mmData;
            }
        }

        Debug.LogError($"{LogPrefix} {BasePrefabPath} has no MotionMatchingStage with its data assigned.");
        return null;
    }

    // --- 4. Bake ------------------------------------------------------------------------------------

    private static bool BakeStep()
    {
        var config = LoadConfig();
        return config != null && StyleTransferConfigEditor.GeneratePoseDatabase(config);
    }

    private static StyleTransferConfig LoadConfig()
    {
        var config = AssetDatabase.LoadAssetAtPath<StyleTransferConfig>(ConfigPath);
        if (config == null) Debug.LogError($"{LogPrefix} No config at {ConfigPath}; run Create Config first.");
        return config;
    }

    // --- 5. Train -----------------------------------------------------------------------------------

    [Serializable]
    private class TrainingLog
    {
        public string config;
        public string startedUtc;
        public double seconds;
        public int epochs;
        public string device;
        public string processorType;
        public string graphicsDeviceName;
        public string unityVersion;
        public bool batchMode;

        // The Python trainer's own summary, as JSON text.
        public string summaryJson;
    }

    private static bool TrainStep()
    {
        var config = LoadConfig();
        if (config == null) return false;

        if (!File.Exists(config.GetPoseDatabasePath()) || !File.Exists(config.GetLabelsPath()))
        {
            Debug.LogError($"{LogPrefix} '{config.name}' has no pose database or labels; run Bake first.", config);
            return false;
        }

        var startedUtc = DateTime.UtcNow.ToString("o");
        var watch = Stopwatch.StartNew();
        var summaryJson = StyleTraining.Run(config);
        var seconds = watch.Elapsed.TotalSeconds;

        if (summaryJson == null)
        {
            Debug.LogError($"{LogPrefix} Training '{config.name}' failed after {seconds:0} s.", config);
            return false;
        }

        var log = new TrainingLog
        {
            config = AssetDatabase.GetAssetPath(config),
            startedUtc = startedUtc,
            seconds = seconds,
            epochs = config.epochs,
            device = config.DeviceName,
            processorType = SystemInfo.processorType,
            graphicsDeviceName = SystemInfo.graphicsDeviceName,
            unityVersion = Application.unityVersion,
            batchMode = Application.isBatchMode,
            summaryJson = summaryJson,
        };

        var logPath = Path.Combine(config.GetAssetPath(), "training_log.json");
        File.WriteAllText(logPath, JsonUtility.ToJson(log, true));
        Debug.Log($"{LogPrefix} Trained '{config.name}' in {seconds / 60:0.0} min; wrote {logPath}.");
        return true;
    }

    // --- 6. MM character ----------------------------------------------------------------------------

    private static bool CreateMmCharacterStep()
    {
        var config = LoadConfig();
        if (config == null) return false;

        var style = GameplayTagConfig.instance.GetTag(StyleTagName(DemoStyleName));
        if (style == null)
        {
            Debug.LogError($"{LogPrefix} No tag '{StyleTagName(DemoStyleName)}'; run Set Up Clips first.");
            return false;
        }

        var basePrefab = AssetDatabase.LoadAssetAtPath<GameObject>(BasePrefabPath);
        if (basePrefab == null)
        {
            Debug.LogError($"{LogPrefix} No prefab at {BasePrefabPath}.");
            return false;
        }

        if (!EnsureVariant(basePrefab, config, style)) return false;
        return EnsureScene();
    }

    /// <summary>
    /// Create the variant, or bring an existing one's stage up to date. An existing variant is
    /// re-saved only when something changed, because every save rewrites its fileIDs.
    /// </summary>
    private static bool EnsureVariant(GameObject basePrefab, StyleTransferConfig config, GameplayTagSO style)
    {
        var exists = AssetDatabase.LoadAssetAtPath<GameObject>(VariantPath) != null;
        if (!exists)
        {
            // A preview scene, so creating the variant leaves whatever scene is open untouched.
            var previewScene = EditorSceneManager.NewPreviewScene();
            try
            {
                var instance = (GameObject)PrefabUtility.InstantiatePrefab(basePrefab, previewScene);
                if (!TryConfigureStage(instance, config, style, out _, out var error))
                {
                    Debug.LogError($"{LogPrefix} {error}");
                    return false;
                }

                PrefabUtility.SaveAsPrefabAsset(instance, VariantPath, out var saved);
                if (!saved)
                {
                    Debug.LogError($"{LogPrefix} Could not save {VariantPath}.");
                    return false;
                }
            }
            finally
            {
                EditorSceneManager.ClosePreviewScene(previewScene);
            }

            Debug.Log($"{LogPrefix} Created {VariantPath}, a variant of {BasePrefabPath} running " +
                      $"{StyleTagName(DemoStyleName)} after Inertialization.");
            return true;
        }

        var contents = PrefabUtility.LoadPrefabContents(VariantPath);
        try
        {
            if (!TryConfigureStage(contents, config, style, out var changed, out var error))
            {
                Debug.LogError($"{LogPrefix} {VariantPath}: {error}");
                return false;
            }

            if (changed)
            {
                PrefabUtility.SaveAsPrefabAsset(contents, VariantPath);
                Debug.Log($"{LogPrefix} Updated the StyleTransferStage on {VariantPath}.");
            }
            else
            {
                Debug.Log($"{LogPrefix} {VariantPath} is already set up.");
            }
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(contents);
        }

        return true;
    }

    /// <summary>
    /// Give the character's synthesizer a <see cref="StyleTransferStage"/> directly after its
    /// <see cref="Inertialization"/>, or point the one it has at this config and style.
    /// </summary>
    private static bool TryConfigureStage(GameObject root, StyleTransferConfig config, GameplayTagSO style,
        out bool changed, out string error)
    {
        changed = false;
        error = null;

        var synthesizer = root.GetComponentInChildren<MotionSynthesisComponent>(true);
        if (synthesizer == null)
        {
            error = $"'{root.name}' has no MotionSynthesisComponent.";
            return false;
        }

        var stage = synthesizer.stages.OfType<StyleTransferStage>().FirstOrDefault();
        if (stage == null)
        {
            var inertialization = synthesizer.stages.FindIndex(s => s is Inertialization);
            if (inertialization < 0)
            {
                error = $"'{root.name}' has no Inertialization stage to place the style stage after.";
                return false;
            }

            stage = new StyleTransferStage();
            synthesizer.stages.Insert(inertialization + 1, stage);
            changed = true;
        }

        if (stage.config != config || stage.style != style ||
            stage.rootMotion != StyleTransferStage.RootMotionMode.Preserve)
        {
            stage.config = config;
            stage.style = style;
            stage.rootMotion = StyleTransferStage.RootMotionMode.Preserve;
            changed = true;
        }

        // The synthesizer belongs to the base prefab, so the edit has to be recorded as an override.
        if (changed) PrefabUtility.RecordPrefabInstancePropertyModifications(synthesizer);
        return true;
    }

    /// <summary>A floor, a camera, a light, the keyboard input and one variant instance. Left alone once it exists.</summary>
    private static bool EnsureScene()
    {
        if (AssetExists(ScenePath))
        {
            Debug.Log($"{LogPrefix} {ScenePath} already exists; left as it is.");
            return true;
        }

        if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
        {
            Debug.LogError($"{LogPrefix} The open scene was not saved; {ScenePath} was not created.");
            return false;
        }

        var variant = AssetDatabase.LoadAssetAtPath<GameObject>(VariantPath);
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        foreach (var rootObject in scene.GetRootGameObjects())
        {
            if (!rootObject.TryGetComponent<Camera>(out _)) continue;

            rootObject.transform.SetPositionAndRotation(new Vector3(0f, 2f, -5f), Quaternion.Euler(15f, 0f, 0f));
        }

        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.localScale = new Vector3(10f, 1f, 10f);

        // Control inputs hear the keyboard only through UserInput's static channel.
        new GameObject(nameof(UserInput), typeof(UserInput));

        var character = (GameObject)PrefabUtility.InstantiatePrefab(variant, scene);
        if (character.GetComponentInChildren<MotionSynthesisControlInput>(true) == null)
        {
            var synthesizer = character.GetComponentInChildren<MotionSynthesisComponent>(true);
            synthesizer.gameObject.AddComponent<AnchoredDirectionControlInput>();
            Debug.Log($"{LogPrefix} {VariantPath} carries no control input; added an AnchoredDirectionControlInput.");
        }

        if (!EditorSceneManager.SaveScene(scene, ScenePath))
        {
            Debug.LogError($"{LogPrefix} Could not save {ScenePath}.");
            return false;
        }

        Debug.Log($"{LogPrefix} Created {ScenePath}.");
        return true;
    }

    // --- Shared -------------------------------------------------------------------------------------

    private static string ProjectRoot => Path.GetDirectoryName(Application.dataPath) ?? string.Empty;

    private static bool AssetExists(string assetPath) =>
        !string.IsNullOrEmpty(AssetDatabase.AssetPathToGUID(assetPath, AssetPathToGUIDOptions.OnlyExistingAssets));

    /// <summary>
    /// Drop what the last file loaded. Managed references go first, so the native unload finds them
    /// unreferenced.
    /// </summary>
    private static void ReleaseUnusedAssets()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        EditorUtility.UnloadUnusedAssetsImmediate();
    }
}
}
