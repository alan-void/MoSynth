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
    /// Each range is trimmed by <see cref="MaximumFramesPrediction"/> at its end and by
    /// <see cref="MaximumFramesHistory"/> at its start: a feature vector encodes the frames around
    /// it, so the edge frames of a range have no valid trajectory and would let the search pick a
    /// pose whose trajectory runs off the tagged region.
    /// </remarks>
    [BurstCompile]
    public struct SetTagBurst : IJob
    {
        [ReadOnly] public int MaximumFramesPrediction; // Frames of lookahead the longest trajectory feature needs
        [ReadOnly] public int MaximumFramesHistory; // Frames of history the furthest-back trajectory sample needs
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
                for (int j = start + MaximumFramesHistory; j < end - MaximumFramesPrediction; j++)
                {
                    TagMask[j] = true;
                }
            }
        }
    }
}