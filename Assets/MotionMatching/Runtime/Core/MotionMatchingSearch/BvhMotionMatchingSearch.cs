using System;
using Unity.Collections;
using Unity.Jobs;

namespace MotionMatching
{
/// <summary>
/// The default search: the same result as <see cref="LinearMotionMatchingSearch"/>, but skips whole
/// runs of frames using a two-level hierarchy of bounding boxes over the feature vectors.
/// </summary>
/// <remarks>
/// Traversal is in <see cref="BVHMotionMatchingSearchBurst"/>. The boxes belong to the
/// <see cref="FeatureSet"/>, so they are not disposed here. See openwiki/motion-matching/search-backends.md.
/// </remarks>
[Serializable]
public class BvhMotionMatchingSearch : MotionMatchingSearch
{
    // Two nesting levels of bounding box, coarse ("large") over groups of the fine ("small") ones.
    private NativeArray<float> _largeBoundingBoxMin;
    private NativeArray<float> _largeBoundingBoxMax;
    private NativeArray<float> _smallBoundingBoxMin;
    private NativeArray<float> _smallBoundingBoxMax;

    public override void Initialize(
        FeatureSet featureSet, NativeArray<bool> tagMask,
        NativeArray<float> featuresWeights
        )
    {
        FeatureSet = featureSet;
        TagMask = tagMask;
        FeatureWeights = featuresWeights;
        featureSet.GetBvhBuffers(out _largeBoundingBoxMin,
            out _largeBoundingBoxMax,
            out _smallBoundingBoxMin,
            out _smallBoundingBoxMax);
        SearchResult = new NativeArray<int>(2, Allocator.Domain);
        SearchResult[0] = 0;
        SearchResult[1] = 0;
    }

    public override int FindBestFrame(NativeArray<float> queryFeature, float currentDistance)
    {
        var job = new BVHMotionMatchingSearchBurst
        {
            Valid = FeatureSet.GetValid(),
            TagMask = TagMask,
            Features = FeatureSet.GetFeatures(),
            QueryFeature = queryFeature,
            FeatureWeights = FeatureWeights,
            FeatureSize = FeatureSet.FeatureSize,
            CurrentDistance = currentDistance,
            
            LargeBoundingBoxMin = _largeBoundingBoxMin,
            LargeBoundingBoxMax = _largeBoundingBoxMax,
            SmallBoundingBoxMin = _smallBoundingBoxMin,
            SmallBoundingBoxMax = _smallBoundingBoxMax,
            
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
