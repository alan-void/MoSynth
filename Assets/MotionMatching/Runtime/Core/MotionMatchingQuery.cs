using System;
using Unity.Collections;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Builds the trajectory half of a query vector: what the control input wants, laid out exactly
/// like a database feature vector so the two compare directly.
/// </summary>
/// <remarks>
/// Shared by <see cref="MotionMatchingStage"/> and the learned matcher so both are asked exactly the
/// same question; the pose half differs between them and so is not here.
/// See openwiki/motion-matching/learned-motion-matching.md.
/// </remarks>
public static class MotionMatchingQuery
{
    /// <summary>
    /// Writes every trajectory feature into <paramref name="query"/> and normalizes them.
    /// </summary>
    /// <param name="mmData">The database whose feature definitions describe the layout.</param>
    /// <param name="featureSet">Its extracted features, which own the offsets and the statistics.</param>
    /// <param name="controlInput">Whatever is steering the character.</param>
    /// <param name="character">The character's transform, which bone channels are resolved against.</param>
    /// <param name="query">The full-length query vector; only the trajectory block is touched.</param>
    /// <param name="featureWeights">
    /// The weights the search will use. Bone channels are masked here, so this is written as well
    /// as read.
    /// </param>
    /// <param name="authoredFeatureWeights">
    /// The unmasked weights, which an active bone channel is restored from.
    /// </param>
    public static void FillTrajectory(MotionMatchingData mmData, FeatureSet featureSet,
        MotionMatchingControlInput controlInput, Transform character, Span<float> query,
        NativeArray<float> featureWeights, NativeArray<float> authoredFeatureWeights)
    {
        for (var i = 0; i < mmData.trajectoryFeatures.Count; i++)
        {
            var featureDef = mmData.trajectoryFeatures[i];
            var featureSize = featureSet.GetTrajectoryFeatureFloatCount(i);
            for (var p = 0; p < featureSet.GetPredictionCount(i); ++p)
            {
                var offset = featureSet.GetTrajectoryFeatureOffset(i, p);
                var feature = query.Slice(offset, featureSize);
                if (featureDef.simulationBone)
                {
                    controlInput.GetTrajectoryFeature(featureDef, p, character, feature);
                }
                else
                {
                    // Bone channels are optional constraints: the control input switches one on by
                    // supplying a target. Off channels are weight-masked to zero so they cannot
                    // affect the search.
                    var active = controlInput.GetBoneTrajectoryFeature(featureDef, p, character, feature);
                    for (var f = 0; f < featureSize; f++)
                    {
                        if (!active) feature[f] = 0f;
                        featureWeights[offset + f] = active ? authoredFeatureWeights[offset + f] : 0f;
                    }
                }
            }
        }

        // The database's trajectory floats are normalized, so the query's must be too.
        featureSet.NormalizeTrajectory(query);
    }
}
}
