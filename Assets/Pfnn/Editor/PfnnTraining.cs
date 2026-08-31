using System;
using AnimationTools;
using Python.Runtime;
using UnityEditor;
using UnityEngine;

namespace Pfnn.Editor
{
/// <summary>
/// Drives <c>pfnn_trainer.train</c> for a config.
/// </summary>
/// <remarks>
/// A class of its own rather than a method on the inspector, because the set of hyperparameters
/// that crosses the boundary is the thing that has to match what the config says — and a second
/// caller (a batch menu item, a headless build step) writing its own kwargs dict is how the two
/// would drift apart.
/// <para>
/// The interpreter runs in this process, synchronously on the main thread.
/// </para>
/// </remarks>
public static class PfnnTraining
{
    /// <summary>
    /// Train <paramref name="config"/>'s network and write its checkpoint. Sets
    /// <see cref="PfnnConfig.hasTrained"/> only when training actually produced one.
    /// </summary>
    public static void Run(PfnnConfig config)
    {
        // A domain reload while the interpreter is mid-call takes the editor down with it, so hold
        // reloads off for the duration.
        EditorApplication.LockReloadAssemblies();
        try
        {
            PythonRuntime.EnsureInitialized();

            // Marshalled into Python as a callable. Training is synchronous on the main thread, so
            // this fires on the main thread too and may touch the editor UI directly.
            Action<string, double> report = (stage, fraction) =>
                EditorUtility.DisplayProgressBar("Training PFNN", stage, (float)fraction);

            using (Py.GIL())
            {
                var trainer = PythonRuntime.Import("pfnn_trainer", reload: true);

                using var args = new PyTuple(new[]
                {
                    config.GetAssetPath().ToPython(),
                    config.name.ToPython(),
                    config.GetCheckpointPath().ToPython(),
                });

                using var kwargs = new PyDict();
                kwargs["excluded_bones"] = PfnnBoneSelection.ToPython(config);
                kwargs["window_radius"] = config.windowRadiusFrames.ToPython();
                kwargs["window_stride"] = config.windowStrideFrames.ToPython();
                kwargs["hidden_units"] = config.hiddenUnits.ToPython();
                kwargs["dropout"] = ((double)config.dropout).ToPython();
                kwargs["epochs"] = config.epochs.ToPython();
                kwargs["batch_size"] = config.batchSize.ToPython();
                kwargs["learning_rate"] = ((double)config.learningRate).ToPython();
                kwargs["weight_decay"] = ((double)config.weightDecay).ToPython();
                kwargs["validation_fraction"] = ((double)config.validationFraction).ToPython();
                kwargs["seed"] = config.seed.ToPython();
                kwargs["device"] = config.DeviceName.ToPython();
                // This config authors no matching features, so there is no .mmfeatures beside the
                // poses to read.
                kwargs["with_features"] = false.ToPython();
                kwargs["progress"] = report.ToPython();

                using var summary = trainer.InvokeMethod("train", args, kwargs);
                Debug.Log($"[PFNN] {summary}");

                // Only on the success path: train() throwing leaves the old checkpoint on disk, and
                // it is no less stale than it was a moment ago.
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
}
}
