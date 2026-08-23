using System;
using Unity.Collections;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Given a query feature vector describing what the character should do next, finds the closest
/// frame in the database. Subclass to add a new search method — brute force, an acceleration
/// structure, a learned index — without touching <see cref="MotionMatchingStage"/>.
/// </summary>
/// <remarks>
/// Inspector-authored via <c>[SerializeReference]</c>, so a concrete search must be
/// <c>[Serializable]</c> with a parameterless constructor. <see cref="Initialize"/> runs once from
/// the stage's Init, <see cref="FindBestFrame"/> once per search tick, <see cref="Dispose"/> at the
/// end.
/// <para>
/// Every implementation must use the same distance metric, <see cref="MotionMatchingStage.SqrDistance"/>,
/// since the caller compares results against it.
/// </para>
/// </remarks>
[Serializable]
public abstract class MotionMatchingSearch
{
    /// <summary>The database being searched: feature vectors, their validity, and their layout.</summary>
    protected FeatureSet FeatureSet;

    /// <summary>One flag per frame; false excludes it. How tag queries narrow the database.</summary>
    [ReadOnly] protected NativeArray<bool> TagMask;

    /// <summary>Per-float weights for the distance metric, length <see cref="FeatureSet.FeatureSize"/>.</summary>
    [ReadOnly] protected NativeArray<float> FeatureWeights;

    /// <summary>Job output, since a Burst job cannot return a value. Element 0 is the best frame.</summary>
    protected NativeArray<int> SearchResult;

    /// <summary>The search used when a stage has not been given one explicitly.</summary>
    public static MotionMatchingSearch Default => new BvhMotionMatchingSearch();

    /// <summary>
    /// Binds to a database and does any one-time setup. Required override despite being virtual —
    /// the base throws. The arrays stay alive for the stage's lifetime, so store rather than copy
    /// them; release anything allocated here in <see cref="Dispose"/>.
    /// </summary>
    public virtual void Initialize(FeatureSet featureSet, NativeArray<bool> tagMask, NativeArray<float> featuresWeights)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Returns the frame closest to <paramref name="queryFeature"/>, considering only frames that
    /// are valid in the <see cref="FeatureSet"/> and allowed by <see cref="TagMask"/>.
    /// </summary>
    /// <param name="currentDistance">
    /// Distance to the frame already playing, or MaxValue when it is unusable. Acts as the initial
    /// best, so candidates worse than it are rejected.
    /// </param>
    /// <returns>The winning frame, or -1 when nothing beats <paramref name="currentDistance"/>.</returns>
    public abstract int FindBestFrame(NativeArray<float> queryFeature, float currentDistance);

    /// <summary>Releases native memory allocated in <see cref="Initialize"/>.</summary>
    public virtual void Dispose()
    {
    }
}
}
