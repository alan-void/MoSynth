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
/// Shared by <see cref="MotionMatchingStage"/> and the learned matcher, and that sharing is the
/// point rather than a convenience. The two methods are meant to be compared against each other on
/// one database, so a second implementation of "what is the character being asked to do" — even one
/// that started as a faithful copy — would make the comparison measure the difference between two
/// query builders as much as the difference between two matchers.
/// <para>
/// The pose half is <em>not</em> here, because that is the one place the two genuinely differ: the
/// classic matcher reads it out of the frame it is playing, while a learned one carries it in the
/// state its networks advance.
/// </para>
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
        // One slice per prediction horizon.
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
