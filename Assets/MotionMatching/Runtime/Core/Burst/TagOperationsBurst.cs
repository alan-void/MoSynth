using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace MotionMatching
{
    /// <summary>Clears any tag filter, making every frame of the database searchable again.</summary>
    [BurstCompile]
    public struct DisableTagBurst : IJob
    {
        [WriteOnly] public NativeArray<bool> TagMask;

        public void Execute()
        {
            for (int i = 0; i < TagMask.Length; i++)
            {
                TagMask[i] = true;
            }
        }
    }

    /// <summary>
    /// Turns a tag query's frame ranges into the per-frame mask the search consumes: true where a
    /// frame is inside some range of the query, false everywhere else.
    /// </summary>
    /// <remarks>
    /// Each range is trimmed by <see cref="MaximumFramesPrediction"/> at its end: a feature vector
    /// encodes the next several frames, so the last frames of a range have no valid trajectory and
    /// would let the search pick a pose whose future runs off the tagged region.
    /// </remarks>
    [BurstCompile]
    public struct SetTagBurst : IJob
    {
        [ReadOnly] public int MaximumFramesPrediction; // Number of prediction frames of the longest trajectory feature
        [ReadOnly] public NativeArray<int> StartRanges;
        [ReadOnly] public NativeArray<int> EndRanges;
        [WriteOnly] public NativeArray<bool> TagMask;

        public void Execute()
        {
            for (int i = 0; i < TagMask.Length; i++)
            {
                TagMask[i] = false;
            }
            for (int i = 0; i < StartRanges.Length; i++)
            {
                int start = StartRanges[i];
                int end = EndRanges[i];
                for (int j = start; j < end - MaximumFramesPrediction; j++)
                {
                    TagMask[j] = true;
                }
            }
        }
    }
}