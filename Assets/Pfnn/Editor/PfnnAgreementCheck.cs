using System;
using System.IO;
using System.Linq;
using AnimationTools;
using Python.Runtime;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace Pfnn.Editor
{
/// <summary>
/// Compares the two definitions of a character-frame pose — <see cref="CharacterSpacePose.Extract"/>
/// in C# and <c>training.training_data.build_training_set</c> in Python — on the same frames of the same
/// database, and reports how far apart they are.
/// </summary>
/// <remarks>
/// A model is trained on one side's arrays and run on the other's, and a disagreement there throws
/// nothing — it only produces bad motion. A menu diagnostic rather than a test because it needs a
/// generated database and an interpreter. Positions and rotations should agree to float precision;
/// rates only to first order, since Python differences consecutive frames while C# reads the
/// pose's velocity channels.
/// </remarks>
public static class PfnnAgreementCheck
{
    /// <summary>Frames sampled across the database. Enough to cover several gait cycles.</summary>
    private const int SampleCount = 64;

    [MenuItem("MoSynth/Pfnn/Check Training Agreement", priority = 300)]
    public static void Run()
    {
        var config = Selection.objects.OfType<PfnnConfig>().FirstOrDefault() ?? FindOnlyConfig();
        if (config == null)
        {
            Debug.LogError("[PFNN] Select a PfnnConfig, or have exactly one in the project.");
            return;
        }

        if (!File.Exists(config.GetPoseDatabasePath()))
        {
            Debug.LogError($"[PFNN] '{config.name}' has no pose database. Press Generate first.");
            return;
        }

        // A domain reload while the interpreter is mid-call takes the editor down with it.
        EditorApplication.LockReloadAssemblies();
        try
        {
            Compare(config);
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

    [MenuItem("MoSynth/Pfnn/Check Training Agreement", validate = true)]
    private static bool CanRun() => !EditorApplication.isPlayingOrWillChangePlaymode;

    private static PfnnConfig FindOnlyConfig()
    {
        var guids = AssetDatabase.FindAssets($"t:{nameof(PfnnConfig)}");
        return guids.Length == 1
            ? AssetDatabase.LoadAssetAtPath<PfnnConfig>(AssetDatabase.GUIDToAssetPath(guids[0]))
            : null;
    }

    private static void Compare(PfnnConfig config)
    {
        var poseSet = config.GetOrImportPoseSet();
        if (poseSet == null)
        {
            config.TryValidate(out var reason);
            Debug.LogError($"[PFNN] could not read '{config.name}' pose database: {reason}");
            return;
        }

        var skeleton = config.Skeleton;
        var skeletonData = skeleton.GetSkeletonData();
        var frameDef = SimulationFrameDef.Default(skeleton);
        var frameTime = poseSet.FrameTime;

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
                frameTime, positions, rotations);

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
            dynamic module = PythonRuntime.Import("pfnn.agreement", reload: true);
            dynamic report = module.compare(config.GetAssetPath(), config.name, frames,
                csharpPositions, csharpRotations, boneCount);

            Debug.Log($"[PFNN] {report}");
        }
    }
}
}
