using System;
using System.IO;
using System.Linq;
using AnimationTools;
using Python.Runtime;
using UnityEditor;
using UnityEngine;

namespace Pfnn.Editor
{
/// <summary>
/// Compares the two definitions of a character-frame pose — <see cref="CharacterSpacePose.Extract"/>
/// in C# and <c>training_data.build_training_set</c> in Python — on the same frames of the same
/// database, and reports how far apart they are.
/// </summary>
/// <remarks>
/// A model is trained on arrays produced by one of these and run on arrays produced by the other.
/// If the two disagree about the reference frame, the units, or the rate convention, nothing
/// throws: the network simply produces bad motion and the cause is invisible from the symptom. Both
/// sides are unit-tested against the same properties, but until this ran, nothing had compared the
/// actual numbers.
/// <para>
/// A diagnostic rather than a test, because it needs a generated database and a working interpreter
/// — neither of which the edit-mode suites are allowed to assume. Positions and rotations should
/// agree to float precision. The rates agree only to first order by construction, so they are
/// reported separately: Python differences consecutive frame-local poses, while C# composes the
/// instantaneous rate implied by a pose's own velocity channels.
/// </para>
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
        using var positions = new Unity.Collections.NativeArray<Unity.Mathematics.float3>(
            boneCount, Unity.Collections.Allocator.Temp);
        using var rotations = new Unity.Collections.NativeArray<Unity.Mathematics.quaternion>(
            boneCount, Unity.Collections.Allocator.Temp);

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
            dynamic module = PythonRuntime.Import("pfnn_agreement", reload: true);
            dynamic report = module.compare(config.GetAssetPath(), config.name, frames,
                csharpPositions, csharpRotations, boneCount);

            Debug.Log($"[PFNN] {report}");
        }
    }
}
}
