using System;
using AnimationTools;
using Python.Runtime;
using UnityEditor;
using UnityEngine;

namespace Lmm.Editor
{
/// <summary>
/// Drives <c>lmm.trainer</c> for a config: every network, or one of the later two on its own.
/// </summary>
/// <remarks>
/// The single place the config's hyperparameters are marshalled, so every caller sends the same
/// kwargs. Runs synchronously on the main thread.
/// </remarks>
public static class LmmTraining
{
    /// <summary>
    /// Train <paramref name="config"/>'s networks and write its checkpoint. Sets
    /// <see cref="LmmConfig.hasTrained"/> only when training actually produced one.
    /// </summary>
    /// <returns>The trainer's summary as JSON, or null when training failed.</returns>
    public static string Run(LmmConfig config) => Execute(config, Fit.Everything);

    /// <summary>
    /// Refit only the stepper, against the latents the checkpoint already carries.
    /// </summary>
    /// <remarks>
    /// Does not set <see cref="LmmConfig.hasTrained"/>: the autoencoder stays as stale or fresh as
    /// it was, and the inspector cannot tell which field an edit moved.
    /// </remarks>
    public static void RunStepper(LmmConfig config) => Execute(config, Fit.Stepper);

    /// <summary>
    /// Refit only the projector, against the latents the checkpoint already carries.
    /// </summary>
    /// <remarks>
    /// Leaves <see cref="LmmConfig.hasTrained"/> alone, as <see cref="RunStepper"/> does.
    /// </remarks>
    public static void RunProjector(LmmConfig config) => Execute(config, Fit.Projector);

    /// <summary>Which networks a run fits. Anything but <see cref="Everything"/> is a refit.</summary>
    private enum Fit
    {
        Everything,
        Stepper,
        Projector,
    }

    /// <summary>Runs one fit; returns the trainer's summary as JSON, or null when it failed.</summary>
    private static string Execute(LmmConfig config, Fit fit)
    {
        if (!config.TryValidate(out var error))
        {
            Debug.LogError($"[LMM] '{config.name}' cannot be trained — {error}", config);
            return null;
        }

        string summaryJson = null;

        // A domain reload while the interpreter is mid-call takes the editor down with it, so hold
        // reloads off for the duration.
        EditorApplication.LockReloadAssemblies();
        try
        {
            PythonRuntime.EnsureInitialized();

            var title = fit switch
            {
                Fit.Stepper => "Fitting the LMM stepper",
                Fit.Projector => "Fitting the LMM projector",
                _ => "Training LMM"
            };

            // Marshalled into Python as a callable. Training is synchronous on the main thread, so
            // this fires on the main thread too and may touch the editor UI directly.
            Action<string, double> report = (stage, fraction) =>
                EditorUtility.DisplayProgressBar(title, stage, (float)fraction);

            using (Py.GIL())
            {
                var trainer = PythonRuntime.Import("lmm.trainer", reload: true);

                using var summary = fit switch
                {
                    Fit.Stepper => FitStepper(config, trainer, report),
                    Fit.Projector => FitProjector(config, trainer, report),
                    _ => TrainEverything(config, trainer, report)
                };
                Debug.Log($"[LMM] {summary}");

                if (fit == Fit.Everything)
                {
                    // Only on success: if train() throws, the old checkpoint on disk is still stale.
                    config.hasTrained = true;
                    EditorUtility.SetDirty(config);
                    AssetDatabase.SaveAssetIfDirty(config);
                }

                summaryJson = PythonRuntime.ToJson(summary);
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

    /// <summary>
    /// Every network, from the database up. Must be called with the GIL held.
    /// </summary>
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
        kwargs["projector"] = (config.trainStepper && config.trainProjector).ToPython();
        kwargs["projector_hidden"] = config.projectorHiddenUnits.ToPython();
        kwargs["projector_iterations"] = config.projectorIterations.ToPython();
        kwargs["projector_patience"] = config.projectorPatience.ToPython();
        kwargs["projector_sigma"] = ((double)config.projectorNoise).ToPython();
        kwargs["projector_max_seconds"] = ((double)config.projectorMaxSeconds).ToPython();
        AddSharedOptimiser(config, kwargs);
        kwargs["progress"] = report.ToPython();

        return trainer.InvokeMethod("train", args, kwargs);
    }

    /// <summary>
    /// The stepper alone, against an existing checkpoint. Must be called with the GIL held.
    /// </summary>
    /// <remarks>
    /// Kwargs are unprefixed because <c>refit_stepper</c> forwards them to <c>fit_stepper</c>.
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
    /// The projector alone, against an existing checkpoint. Must be called with the GIL held.
    /// </summary>
    /// <remarks>
    /// Kwargs are unprefixed because <c>refit_projector</c> forwards them to <c>fit_projector</c>.
    /// </remarks>
    private static PyObject FitProjector(LmmConfig config, dynamic trainer,
        Action<string, double> report)
    {
        using var args = new PyTuple(new[]
        {
            config.GetCheckpointPath().ToPython(),
            config.mmData.GetAssetPath().ToPython(),
            config.mmData.name.ToPython(),
        });

        using var kwargs = new PyDict();
        kwargs["hidden_units"] = config.projectorHiddenUnits.ToPython();
        kwargs["iterations"] = config.projectorIterations.ToPython();
        kwargs["patience"] = config.projectorPatience.ToPython();
        kwargs["sigma"] = ((double)config.projectorNoise).ToPython();
        kwargs["max_seconds"] = ((double)config.projectorMaxSeconds).ToPython();
        AddSharedOptimiser(config, kwargs);
        kwargs["progress"] = report.ToPython();

        return trainer.InvokeMethod("refit_projector", args, kwargs);
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
    /// Sent unexpanded so Python expands them independently, and the stage can catch the two sides
    /// disagreeing.
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
