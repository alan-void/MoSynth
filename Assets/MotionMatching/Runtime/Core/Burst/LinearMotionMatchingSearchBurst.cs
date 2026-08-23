using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace MotionMatching
{
    /// <summary>
    /// Brute-force scan for the closest feature vector: scores every valid, unmasked frame and keeps
    /// the best. Seeded with <see cref="CurrentDistance"/>, so it reports -1 when nothing beats the
    /// frame already playing.
    /// </summary>
    /// <remarks>
    /// Unlike <see cref="BVHMotionMatchingSearchBurst"/> the inner dimension loop has no early exit
    /// once the running distance passes the best so far, so every candidate is scored in full.
    /// </remarks>
    [BurstCompile]
    public struct LinearMotionMatchingSearchBurst : IJob
    {
        [ReadOnly] public NativeArray<bool> Valid;
        [ReadOnly] public NativeArray<bool> TagMask; // TODO: convert to a bitmask to optimize memory
        [ReadOnly] public NativeArray<float> Features;
        [ReadOnly] public NativeArray<float> QueryFeature;
        [ReadOnly] public NativeArray<float> FeatureWeights;
        [ReadOnly] public int FeatureSize;
        [ReadOnly] public float CurrentDistance;

        [WriteOnly] public NativeArray<int> BestIndex;

        public void Execute()
        {
            float minDistance = CurrentDistance;
            int bestIndex = -1;

            for (int i = 0; i < Valid.Length; ++i)
            {
                if (Valid[i] && TagMask[i])
                {
                    float sqrDistance = 0.0f;
                    int featureIndex = i * FeatureSize;

                    for (int j = 0; j < FeatureSize; ++j)
                    {
                        float diff = Features[featureIndex + j] - QueryFeature[j];
                        sqrDistance += diff * diff * FeatureWeights[j];
                    }

                    if (sqrDistance < minDistance)
                    {
                        minDistance = sqrDistance;
                        bestIndex = i;
                    }
                }
            }

            BestIndex[0] = bestIndex;
        }
    }

}