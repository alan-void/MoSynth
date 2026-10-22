using System;
using System.Collections.Generic;
using AnimationTools;
using Python.Runtime;
using UnityEditor;
using UnityEngine;

namespace Style.Editor
{
/// <summary>
/// Drives <c>style.trainer.train</c> for a config.
/// </summary>
/// <remarks>
/// The single place the config's hyperparameters are marshalled, so every caller sends the same
/// kwargs. Runs synchronously on the main thread.
/// </remarks>
public static class StyleTraining
{
    /// <summary>
    /// Train <paramref name="config"/>'s model and write its checkpoint. Sets
    /// <see cref="StyleTransferConfig.hasTrained"/> only when training actually produced one.
    /// </summary>
    /// <returns>The trainer's summary as JSON, or null when training failed.</returns>
    public static string Run(StyleTransferConfig config)
    {
        if (config.neutralStyle == null)
        {
            Debug.LogError($"[Style] '{config.name}': assign a Neutral Style before training.", config);
            return null;
        }

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
                EditorUtility.DisplayProgressBar("Training Style Model", stage, (float)fraction);

            using (Py.GIL())
            {
                var trainer = PythonRuntime.Import("style.trainer", reload: true);

                using var args = new PyTuple(new[]
                {
                    config.GetAssetPath().ToPython(),
                    config.name.ToPython(),
                    config.GetCheckpointPath().ToPython(),
                });

                using var kwargs = new PyDict();
                kwargs["neutral_style"] = config.neutralStyle.TagName.ToPython();
                kwargs["excluded_styles"] = StringList(config.ExcludedStyleNames());
                kwargs["excluded_bones"] = PredictedBoneSelection.ToPython(config.excludedBones);
                kwargs["hidden_units"] = config.hiddenUnits.ToPython();
                kwargs["layers"] = config.layers.ToPython();
                kwargs["embed_dim"] = config.embedDim.ToPython();
                kwargs["history_offsets"] = FloatList(config.historyOffsets);
                kwargs["gait_sigma"] = ((double)config.gaitSigma).ToPython();
                kwargs["epochs"] = config.epochs.ToPython();
                kwargs["batch_size"] = config.batchSize.ToPython();
                kwargs["learning_rate"] = ((double)config.learningRate).ToPython();
                kwargs["weight_decay"] = ((double)config.weightDecay).ToPython();
                kwargs["validation_fraction"] = ((double)config.validationFraction).ToPython();
                kwargs["seed"] = config.seed.ToPython();
                kwargs["device"] = config.DeviceName.ToPython();
                kwargs["progress"] = report.ToPython();

                using var summary = trainer.InvokeMethod("train", args, kwargs);
                summaryJson = PythonRuntime.ToJson((PyObject)summary);

                // Training cost is reported, so the time goes into the log beside the machine.
                Debug.Log($"[Style] Trained '{config.name}' in {SummarySeconds((PyObject)summary)} on " +
                          $"{SystemInfo.processorType} / {SystemInfo.graphicsDeviceName}, " +
                          $"{config.epochs} epochs, in the Editor. Summary: {summaryJson}");

                // Only on success: if train() throws, the old checkpoint on disk is still stale.
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

        return summaryJson;
    }

    /// <summary>The summary's <c>seconds</c> entry, formatted; "unknown time" when it has none.</summary>
    private static string SummarySeconds(PyObject summary)
    {
        using var key = "seconds".ToPython();
        using var seconds = summary.InvokeMethod("get", key);
        return seconds.IsNone() ? "unknown time" : $"{seconds.As<double>():0.#} s";
    }

    /// <summary>Must be called with the GIL held.</summary>
    private static PyList StringList(IEnumerable<string> values)
    {
        var list = new PyList();
        foreach (var value in values)
        {
            using var item = value.ToPython();
            list.Append(item);
        }

        return list;
    }

    /// <summary>Must be called with the GIL held.</summary>
    private static PyList FloatList(IEnumerable<float> values)
    {
        var list = new PyList();
        foreach (var value in values)
        {
            using var item = ((double)value).ToPython();
            list.Append(item);
        }

        return list;
    }
}
}
