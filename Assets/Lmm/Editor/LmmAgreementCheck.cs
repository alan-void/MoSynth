using System;
using System.IO;
using System.Linq;
using AnimationTools;
using Python.Runtime;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace Lmm.Editor
{
/// <summary>
/// Three diagnostics that need a generated database and a working interpreter, so none of them can
/// be an edit-mode test: whether the two definitions of a character-frame pose agree, how well a
/// trained checkpoint reconstructs the database it learned, and how far its stepper wanders when
/// nothing corrects it.
/// </summary>
/// <remarks>
/// The agreement half is <c>PfnnAgreementCheck</c>'s, pointed at a <c>MotionMatchingData</c>
/// instead of a <c>PfnnConfig</c> — it compares <see cref="CharacterSpacePose.Extract"/> against
/// <c>training_data.build_training_set</c> on the same frames. A model is trained on arrays
/// produced by one of those and run on arrays produced by the other, and a disagreement about the
/// reference frame, the units or the rate convention does not throw: the network simply produces
/// bad motion and the cause is invisible from the symptom.
/// <para>
/// The reconstruction half answers the question the training loss cannot. That loss is computed on
/// standardised vectors with per-block weights, so it is comparable between runs and comparable to
/// nothing else — it cannot say whether a foot is a centimetre or a hand's breadth out of place.
/// </para>
/// <para>
/// The drift report answers what neither of the other two can reach: both score a single frame
/// against its own database row, and a stepper fails by accumulating error over many frames.
/// </para>
/// </remarks>
public static class LmmAgreementCheck
{
    /// <summary>Frames sampled across the database. Enough to cover several gait cycles.</summary>
    private const int SampleCount = 64;

    [MenuItem("MoSynth/Lmm/Check Training Agreement", priority = 300)]
    public static void CheckAgreement() => Run(config =>
    {
        var mmData = config.mmData;
        var poseSet = mmData.GetOrImportPoseSet();
        if (poseSet == null)
        {
            mmData.TryValidate(out var reason);
            Debug.LogError($"[LMM] could not read '{mmData.name}' pose database: {reason}");
            return;
        }

        Compare(mmData, poseSet);
    });

    [MenuItem("MoSynth/Lmm/Report Reconstruction Error", priority = 301)]
    public static void CheckReconstruction() => Report((runtime, policy, set, config) =>
        runtime.reconstruction_report(policy, set, ReportFrames, config.validationFraction));

    /// <summary>
    /// How far a free-running stepper wanders from the database, and what that costs in metres.
    /// </summary>
    /// <remarks>
    /// The reconstruction report cannot answer this. It scores a pose against the latent the
    /// compressor baked for that very frame, while between searches the stage feeds the
    /// decompressor a latent the stepper produced — and the failure mode of a stepper is error
    /// that compounds, which nothing measured one frame at a time can see.
    /// </remarks>
    [MenuItem("MoSynth/Lmm/Report Stepper Drift", priority = 302)]
    public static void CheckStepperDrift() => Report((runtime, policy, set, config) =>
    {
        if (!(bool)policy.has_stepper())
        {
            Debug.LogError($"[LMM] '{config.name}' carries no stepper, so there is nothing to run " +
                           "free. Press Fit Stepper Only on the config.", config);
            return;
        }

        runtime.rollout_report(policy, set, RolloutSeeds, config.validationFraction);
    });

    /// <summary>
    /// States a drift report runs from. Each one is rolled the full horizon, so this is thirty
    /// forward passes apiece — enough to be representative, quick enough for a menu item.
    /// </summary>
    private const int RolloutSeeds = 2048;

    /// <summary>
    /// Loads the checkpoint and its database once, then hands both to <paramref name="report"/>.
    /// </summary>
    /// <remarks>
    /// Shared because the loading is the slow part — a few hundred megabytes of poses — and
    /// because a second copy of it would be a second place for the database and the checkpoint to
    /// be resolved differently.
    /// </remarks>
    private static void Report(Action<dynamic, dynamic, dynamic, LmmConfig> report) => Run(config =>
    {
        if (!File.Exists(config.GetCheckpointPath()))
        {
            Debug.LogError($"[LMM] '{config.name}' has not been trained yet.", config);
            return;
        }

        PythonRuntime.EnsureInitialized();
        using (Py.GIL())
        {
            dynamic runtime = PythonRuntime.Import("lmm_runtime", reload: true);
            dynamic trainingData = PythonRuntime.Import("training_data", reload: true);

            dynamic policy = runtime.LmmPolicy(config.GetCheckpointPath());
            dynamic set = trainingData.load_database(config.mmData.GetAssetPath(),
                config.mmData.name);

            Debug.Log($"[LMM] {config.name}: {(string)policy.describe()}");
            report(runtime, policy, set, config);
        }
    });

    /// <summary>
    /// Frames the reconstruction report scores, evenly spread. Enough to be representative without
    /// making an interactive check take minutes on a database of a few hundred thousand.
    /// </summary>
    /// <remarks>
    /// They are drawn from the held-out tail the config trained against, not from the whole
    /// database, so the number answers how well the model generalises rather than how well it
    /// memorised — which on a database this size are very different questions.
    /// </remarks>
    private const int ReportFrames = 8192;

    [MenuItem("MoSynth/Lmm/Check Training Agreement", validate = true)]
    [MenuItem("MoSynth/Lmm/Report Reconstruction Error", validate = true)]
    [MenuItem("MoSynth/Lmm/Report Stepper Drift", validate = true)]
    private static bool CanRun() => !EditorApplication.isPlayingOrWillChangePlaymode;

    /// <summary>Resolves the config to act on, then runs <paramref name="action"/> safely.</summary>
    private static void Run(Action<LmmConfig> action)
    {
        var config = Selection.objects.OfType<LmmConfig>().FirstOrDefault() ?? FindOnlyConfig();
        if (config == null)
        {
            Debug.LogError("[LMM] Select an LmmConfig, or have exactly one in the project.");
            return;
        }

        if (!config.TryValidate(out var error))
        {
            Debug.LogError($"[LMM] '{config.name}' is not usable — {error}", config);
            return;
        }

        // A domain reload while the interpreter is mid-call takes the editor down with it.
        EditorApplication.LockReloadAssemblies();
        try
        {
            action(config);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
        }
        finally
        {
            EditorApplication.UnlockReloadAssemblies();
        }
    }

    private static LmmConfig FindOnlyConfig()
    {
        var guids = AssetDatabase.FindAssets($"t:{nameof(LmmConfig)}");
        return guids.Length == 1
            ? AssetDatabase.LoadAssetAtPath<LmmConfig>(AssetDatabase.GUIDToAssetPath(guids[0]))
            : null;
    }

    /// <summary>
    /// Extracts the same frames both ways and hands the C# answer to Python to be scored.
    /// </summary>
    private static void Compare(MotionMatching.MotionMatchingData mmData, PoseSet poseSet)
    {
        var skeleton = mmData.Skeleton;
        var skeletonData = skeleton.GetSkeletonData();
        var frameDef = SimulationFrameDef.Default(skeleton);
        var boneCount = skeleton.BoneCount;

        using var positions = new NativeArray<float3>(boneCount, Allocator.Temp);
        using var rotations = new NativeArray<quaternion>(boneCount, Allocator.Temp);

        var step = Mathf.Max(1, poseSet.NumberPoses / SampleCount);
        var frames = new int[Mathf.Min(SampleCount, poseSet.NumberPoses / Mathf.Max(1, step))];
        for (var i = 0; i < frames.Length; i++) frames[i] = i * step;

        var csharpPositions = new float[frames.Length * boneCount * 3];
        var csharpRotations = new float[frames.Length * boneCount * 4];

        for (var f = 0; f < frames.Length; f++)
        {
            CharacterSpacePose.Extract(poseSet.GetPoseBuffer(frames[f]), skeletonData, frameDef,
                poseSet.FrameTime, positions, rotations);

            for (var bone = 0; bone < boneCount; bone++)
            {
                var p = (f * boneCount + bone) * 3;
                csharpPositions[p] = positions[bone].x;
                csharpPositions[p + 1] = positions[bone].y;
                csharpPositions[p + 2] = positions[bone].z;

                var r = (f * boneCount + bone) * 4;
                csharpRotations[r] = rotations[bone].value.x;
                csharpRotations[r + 1] = rotations[bone].value.y;
                csharpRotations[r + 2] = rotations[bone].value.z;
                csharpRotations[r + 3] = rotations[bone].value.w;
            }
        }

        PythonRuntime.EnsureInitialized();
        using (Py.GIL())
        {
            dynamic module = PythonRuntime.Import("pfnn_agreement", reload: true);
            Debug.Log($"[LMM] {module.compare(mmData.GetAssetPath(), mmData.name, frames, csharpPositions, csharpRotations, boneCount)}");
        }
    }
}
}
