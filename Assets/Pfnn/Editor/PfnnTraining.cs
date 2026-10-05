using System;
using AnimationTools;
using Python.Runtime;
using UnityEditor;
using UnityEngine;

namespace Pfnn.Editor
{
/// <summary>
/// Drives <c>pfnn.trainer.train</c> for a config.
/// </summary>
/// <remarks>
/// The single place the config's hyperparameters are marshalled, so every caller sends the same
/// kwargs. Runs synchronously on the main thread.
/// </remarks>
public static class PfnnTraining
{
    /// <summary>
    /// Train <paramref name="config"/>'s network and write its checkpoint. Sets
    /// <see cref="PfnnConfig.hasTrained"/> only when training actually produced one.
    /// </summary>
    /// <returns>The trainer's summary as JSON, or null when training failed.</returns>
    public static string Run(PfnnConfig config)
    {
        string summaryJson = null;

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
                var trainer = PythonRuntime.Import("pfnn.trainer", reload: true);

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
                // A PFNN config authors no matching features, so there is no .mmfeatures to read.
                kwargs["with_features"] = false.ToPython();
                kwargs["progress"] = report.ToPython();

                using var summary = trainer.InvokeMethod("train", args, kwargs);
                Debug.Log($"[PFNN] {summary}");

                // Only on success: if train() throws, the old checkpoint on disk is still stale.
                config.hasTrained = true;
                EditorUtility.SetDirty(config);
                AssetDatabase.SaveAssetIfDirty(config);
                summaryJson = PythonRuntime.ToJson((PyObject)summary);
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

        return summaryJson;
    }
}
}
