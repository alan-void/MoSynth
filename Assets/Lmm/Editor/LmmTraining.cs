using System;
using AnimationTools;
using Python.Runtime;
using UnityEditor;
using UnityEngine;

namespace Lmm.Editor
{
/// <summary>
/// Drives <c>lmm_trainer</c> for a config, either whole or stepper-only.
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
public static class LmmTraining
{
    /// <summary>
    /// Train <paramref name="config"/>'s networks and write its checkpoint. Sets
    /// <see cref="LmmConfig.hasTrained"/> only when training actually produced one.
    /// </summary>
    public static void Run(LmmConfig config) => Execute(config, stepperOnly: false);

    /// <summary>
    /// Refit only the stepper, against the latents the checkpoint already carries.
    /// </summary>
    /// <remarks>
    /// Deliberately does <em>not</em> set <see cref="LmmConfig.hasTrained"/>: it leaves the
    /// autoencoder exactly as stale or as fresh as it found it, and the inspector cannot tell
    /// which field an edit moved. Saying the checkpoint is up to date would be a guess.
    /// </remarks>
    public static void RunStepper(LmmConfig config) => Execute(config, stepperOnly: true);

    private static void Execute(LmmConfig config, bool stepperOnly)
    {
        if (!config.TryValidate(out var error))
        {
            Debug.LogError($"[LMM] '{config.name}' cannot be trained — {error}", config);
            return;
        }

        // A domain reload while the interpreter is mid-call takes the editor down with it, so hold
        // reloads off for the duration.
        EditorApplication.LockReloadAssemblies();
        try
        {
            PythonRuntime.EnsureInitialized();

            var title = stepperOnly ? "Fitting the LMM stepper" : "Training LMM";

            // Marshalled into Python as a callable. Training is synchronous on the main thread, so
            // this fires on the main thread too and may touch the editor UI directly.
            Action<string, double> report = (stage, fraction) =>
                EditorUtility.DisplayProgressBar(title, stage, (float)fraction);

            using (Py.GIL())
            {
                var trainer = PythonRuntime.Import("lmm_trainer", reload: true);

                using var summary = stepperOnly
                    ? FitStepper(config, trainer, report)
                    : TrainEverything(config, trainer, report);
                Debug.Log($"[LMM] {summary}");

                if (!stepperOnly)
                {
                    // Only on the success path: train() throwing leaves the old checkpoint on disk,
                    // and it is no less stale than it was a moment ago.
                    config.hasTrained = true;
                    EditorUtility.SetDirty(config);
                    AssetDatabase.SaveAssetIfDirty(config);
                }
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
    /// Every network, from the database up. Must be called with the GIL held.
    /// </summary>
    /// <remarks>
    /// The database belongs to the <c>MotionMatchingData</c>; only the checkpoint is this config's
    /// own. That split is the point of the asset: it learns someone else's database rather than
    /// owning one.
    /// </remarks>
    private static PyObject TrainEverything(LmmConfig config, dynamic trainer,
        Action<string, double> report)
    {
        using var args = new PyTuple(new[]
        {
            config.mmData.GetAssetPath().ToPython(),
            config.mmData.name.ToPython(),
            config.GetCheckpointPath().ToPython(),
        });

        using var kwargs = new PyDict();
        kwargs["excluded_bones"] = PredictedBoneSelection.ToPython(config.excludedBones);
        kwargs["feature_weights"] = AuthoredWeights(config);
        kwargs["latent_size"] = config.latentSize.ToPython();
        kwargs["latent_velocity_weight"] = ((double)config.latentVelocityWeight).ToPython();
        kwargs["compressor_hidden"] = config.compressorHiddenUnits.ToPython();
        kwargs["decompressor_hidden"] = config.decompressorHiddenUnits.ToPython();
        kwargs["iterations"] = config.iterations.ToPython();
        kwargs["max_seconds"] = ((double)config.maxSeconds).ToPython();
        kwargs["stepper"] = config.trainStepper.ToPython();
        kwargs["stepper_hidden"] = config.stepperHiddenUnits.ToPython();
        kwargs["stepper_window"] = config.stepperWindow.ToPython();
        kwargs["stepper_iterations"] = config.stepperIterations.ToPython();
        kwargs["stepper_patience"] = config.stepperPatience.ToPython();
        kwargs["stepper_max_seconds"] = ((double)config.stepperMaxSeconds).ToPython();
        AddSharedOptimiser(config, kwargs);
        kwargs["progress"] = report.ToPython();

        return trainer.InvokeMethod("train", args, kwargs);
    }

    /// <summary>
    /// The stepper alone, against an existing checkpoint. Must be called with the GIL held.
    /// </summary>
    /// <remarks>
    /// The stepper's own hyperparameters lose their prefix here, because
    /// <c>refit_stepper</c> forwards them straight to <c>fit_stepper</c> — on that side there is
    /// only one network and nothing to disambiguate them from.
    /// </remarks>
    private static PyObject FitStepper(LmmConfig config, dynamic trainer,
        Action<string, double> report)
    {
        using var args = new PyTuple(new[]
        {
            config.GetCheckpointPath().ToPython(),
            config.mmData.GetAssetPath().ToPython(),
            config.mmData.name.ToPython(),
        });

        using var kwargs = new PyDict();
        kwargs["hidden_units"] = config.stepperHiddenUnits.ToPython();
        kwargs["window"] = config.stepperWindow.ToPython();
        kwargs["iterations"] = config.stepperIterations.ToPython();
        kwargs["patience"] = config.stepperPatience.ToPython();
        kwargs["max_seconds"] = ((double)config.stepperMaxSeconds).ToPython();
        AddSharedOptimiser(config, kwargs);
        kwargs["progress"] = report.ToPython();

        return trainer.InvokeMethod("refit_stepper", args, kwargs);
    }

    /// <summary>
    /// The optimiser settings both fits share, under the names both Python entry points read.
    /// </summary>
    private static void AddSharedOptimiser(LmmConfig config, PyDict kwargs)
    {
        kwargs["batch_size"] = config.batchSize.ToPython();
        kwargs["learning_rate"] = ((double)config.learningRate).ToPython();
        kwargs["weight_decay"] = ((double)config.weightDecay).ToPython();
        kwargs["learning_rate_decay"] = ((double)config.learningRateDecay).ToPython();
        kwargs["validation_fraction"] = ((double)config.validationFraction).ToPython();
        kwargs["validation_interval"] = config.validationInterval.ToPython();
        kwargs["seed"] = config.seed.ToPython();
        kwargs["device"] = config.DeviceName.ToPython();
    }

    /// <summary>
    /// The authored search weights, one per feature definition, as a Python list. Must be called
    /// with the GIL held.
    /// </summary>
    /// <remarks>
    /// Sent per definition rather than per float because that is how they are authored, and Python
    /// expands them against the same schema the C# side does. Sending the expansion instead would
    /// mean the two sides could not disagree — and could also not notice that they had.
    /// </remarks>
    private static PyList AuthoredWeights(LmmConfig config)
    {
        var list = new PyList();
        for (var i = 0; i < config.FeatureDefinitionCount; i++)
        {
            var weight = i < config.featureWeights.Count ? config.featureWeights[i] : 1f;
            using var value = ((double)weight).ToPython();
            list.Append(value);
        }

        return list;
    }
}
}
