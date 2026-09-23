using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;

namespace MotionMatching
{
    /// <summary>
    /// How many consecutive database frames each level of bounding box covers. Boxes group frames
    /// by index, not by proximity in feature space, so box <c>k</c> of a level spans frames
    /// <c>[k * size, (k + 1) * size)</c> and box membership is pure integer division.
    /// </summary>
    public static class BVHConsts
    {
        public static readonly int LargeBVHSize = 64;
        public static readonly int SmallBVHSize = 16;
    }

    /// <summary>
    /// Builds the acceleration structure: for each run of frames, the per-dimension min and max of
    /// their feature vectors. Run once when the <see cref="FeatureSet"/> is prepared.
    /// </summary>
    /// <remarks>
    /// Small boxes nest inside large ones only because <see cref="BVHConsts.LargeBVHSize"/> is a
    /// multiple of <see cref="BVHConsts.SmallBVHSize"/>. The search relies on that; keep it true.
    /// </remarks>
    [BurstCompile]
    public struct BVHMotionMatchingComputeBounds : IJob
    {
        [ReadOnly] public NativeArray<float> Features;
        [ReadOnly] public int FeatureSize;
        [ReadOnly] public int NumberBoundingBoxLarge;
        [ReadOnly] public int NumberBoundingBoxSmall;

        public NativeArray<float> LargeBoundingBoxMin; // Size = NumberBoundingBoxLarge x FeatureSize
        public NativeArray<float> LargeBoundingBoxMax; // Size = NumberBoundingBoxLarge x FeatureSize
        public NativeArray<float> SmallBoundingBoxMin; // Size = NumberBoundingBoxSmall x FeatureSize
        public NativeArray<float> SmallBoundingBoxMax; // Size = NumberBoundingBoxSmall x FeatureSize

        public void Execute()
        {
            var largeBoxSize = BVHConsts.LargeBVHSize;
            var smallBoxSize = BVHConsts.SmallBVHSize;

            var numberFrames = Features.Length / FeatureSize;

            for (int i = 0; i < LargeBoundingBoxMin.Length; i++) LargeBoundingBoxMin[i] = float.MaxValue;
            for (int i = 0; i < LargeBoundingBoxMax.Length; i++) LargeBoundingBoxMax[i] = float.MinValue;
            for (int i = 0; i < SmallBoundingBoxMin.Length; i++) SmallBoundingBoxMin[i] = float.MaxValue;
            for (int i = 0; i < SmallBoundingBoxMax.Length; i++) SmallBoundingBoxMax[i] = float.MinValue;

            for (int i = 0; i < numberFrames; ++i)
            {
                int iSmall = i / smallBoxSize;
                int iSmallIndex = iSmall * FeatureSize;
                int iLarge = i / largeBoxSize;
                int iLargeIndex = iLarge * FeatureSize;

                for (int j = 0; j < FeatureSize; ++j)
                {
                    float feature = Features[i * FeatureSize + j];
                    LargeBoundingBoxMin[iLargeIndex + j] = math.min(LargeBoundingBoxMin[iLargeIndex + j], feature);
                    LargeBoundingBoxMax[iLargeIndex + j] = math.max(LargeBoundingBoxMax[iLargeIndex + j], feature);
                    SmallBoundingBoxMin[iSmallIndex + j] = math.min(SmallBoundingBoxMin[iSmallIndex + j], feature);
                    SmallBoundingBoxMax[iSmallIndex + j] = math.max(SmallBoundingBoxMax[iSmallIndex + j], feature);
                }
            }
        }
    }

    /// <summary>
    /// Finds the closest database frame to a query vector, using the bounding boxes to skip runs of
    /// frames wholesale.
    /// </summary>
    /// <remarks>
    /// The distance to a box's nearest point is a lower bound for every frame inside it, so a box that
    /// already reaches the best distance is skipped whole; dimension loops break early for the same
    /// reason. <see cref="CurrentDistance"/> seeds the best, so a query nothing beats leaves
    /// <see cref="BestIndex"/> at -1.
    /// </remarks>
    [BurstCompile]
    public struct BVHMotionMatchingSearchBurst : IJob
    {
        [ReadOnly] public NativeArray<bool> Valid; // TODO: If all features are valid, this will be unnecessary
        [ReadOnly] public NativeArray<bool> TagMask; // TODO: convert to a bitmask to optimize memory
        [ReadOnly] public NativeArray<float> Features;
        [ReadOnly] public NativeArray<float> QueryFeature;
        [ReadOnly] public NativeArray<float> FeatureWeights; // Size = FeatureSize
        [ReadOnly] public int FeatureSize;
        [ReadOnly] public float CurrentDistance;
        [ReadOnly] public NativeArray<float> LargeBoundingBoxMin; // Size = NumberBoundingBoxLarge x FeatureSize
        [ReadOnly] public NativeArray<float> LargeBoundingBoxMax; // Size = NumberBoundingBoxLarge x FeatureSize
        [ReadOnly] public NativeArray<float> SmallBoundingBoxMin; // Size = NumberBoundingBoxSmall x FeatureSize
        [ReadOnly] public NativeArray<float> SmallBoundingBoxMax; // Size = NumberBoundingBoxSmall x FeatureSize

        [WriteOnly] public NativeArray<int> BestIndex;

        public void Execute()
        {
            var largeBoxSize = BVHConsts.LargeBVHSize;
            var smallBoxSize = BVHConsts.SmallBVHSize;

            float min = CurrentDistance;
            int bestIndex = -1;
            const int startIndex = 0;
            int endIndex = Valid.Length;
            int i = startIndex;
            while (i < endIndex)
            {
                int iLarge = i / largeBoxSize;
                int iLargeIndex = iLarge * FeatureSize;
                int iLargeNext = (iLarge + 1) * largeBoxSize;

                float currentCost = 0.0f;
                for (int j = 0; j < FeatureSize; ++j)
                {
                    float query = QueryFeature[j];
                    float cost = query - math.clamp(query, LargeBoundingBoxMin[iLargeIndex + j], LargeBoundingBoxMax[iLargeIndex + j]);
                    currentCost += cost * cost * FeatureWeights[j];
                    if (currentCost >= min)
                    {
                        break;
                    }
                }

                if (currentCost >= min)
                {
                    i = iLargeNext;
                    continue;
                }

                while (i < iLargeNext && i < endIndex)
                {
                    int iSmall = i / smallBoxSize;
                    int iSmallIndex = iSmall * FeatureSize;
                    int iSmallNext = (iSmall + 1) * smallBoxSize;

                    currentCost = 0.0f;
                    for (int j = 0; j < FeatureSize; ++j)
                    {
                        float query = QueryFeature[j];
                        float cost = query - math.clamp(query, SmallBoundingBoxMin[iSmallIndex + j], SmallBoundingBoxMax[iSmallIndex + j]);
                        currentCost += cost * cost * FeatureWeights[j];
                        if (currentCost >= min)
                        {
                            break;
                        }
                    }

                    if (currentCost >= min)
                    {
                        i = iSmallNext;
                        continue;
                    }

                    while (i < iSmallNext && i < endIndex)
                    {
                        if (!Valid[i] || !TagMask[i]) // TODO: build tag mask for large boxes as well
                        {
                            i += 1;
                            continue;
                        }

                        currentCost = 0.0f;
                        for (int j = 0; j < FeatureSize; ++j)
                        {
                            float query = QueryFeature[j];
                            float cost = query - Features[i * FeatureSize + j];
                            currentCost += cost * cost * FeatureWeights[j];
                            if (currentCost >= min)
                            {
                                break;
                            }
                        }

                        if (currentCost < min)
                        {
                            bestIndex = i;
                            min = currentCost;
                        }

                        i += 1;
                    }
                }
            }
            BestIndex[0] = bestIndex;
        }
    }
}