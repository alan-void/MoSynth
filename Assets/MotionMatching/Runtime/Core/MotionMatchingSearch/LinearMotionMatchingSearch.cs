using System;
using Unity.Collections;
using UnityEngine;
using Unity.Jobs;

namespace MotionMatching
{
/// <summary>
/// Brute force: scores every valid frame. Slower, but with no acceleration structure to get wrong,
/// which makes it the reference to check <see cref="BvhMotionMatchingSearch"/> against when a search
/// starts returning surprising frames.
/// </summary>
[Serializable]
public class LinearMotionMatchingSearch : MotionMatchingSearch
{
    public override void Initialize(FeatureSet featureSet, NativeArray<bool> tagMask, NativeArray<float> featuresWeights)
    {
        FeatureSet = featureSet;
        TagMask = tagMask;
        FeatureWeights = featuresWeights;
        
        SearchResult = new NativeArray<int>(2, Allocator.Domain);
        SearchResult[0] = 0;
        SearchResult[1] = 0;
    }

    public override int FindBestFrame(NativeArray<float> queryFeature, float currentDistance)
    {
        var job = new LinearMotionMatchingSearchBurst
        {
            Valid = FeatureSet.GetValid(),
            TagMask = TagMask,
            Features = FeatureSet.GetFeatures(),
            QueryFeature = queryFeature,
            FeatureWeights = FeatureWeights,
            FeatureSize = FeatureSet.FeatureSize,
            CurrentDistance = currentDistance,
            
            BestIndex = SearchResult
        };
        job.Schedule().Complete();

        return SearchResult[0];
    }

    public override void Dispose()
    {
        if (SearchResult.IsCreated) SearchResult.Dispose();
    }
}
}