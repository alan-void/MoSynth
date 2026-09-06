using System;
using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Picks keyframes by where they are drawn, not by where they are in time, so hit areas stay
    /// the same size at every zoom.
    /// </summary>
    public static class TimelineKeyHitTester
    {
        /// <summary>How far from a key's centre a click still counts, in pixels.</summary>
        public const float PickTolerance = 6f;

        /// <summary>
        /// The frame of the key nearest <paramref name="x"/> within tolerance, or -1. Ties go to the
        /// earlier key, so a click between two touching keys is not decided by list order.
        /// </summary>
        public static int Pick(IReadOnlyList<int> frames, float x, Func<int, float> frameToX)
        {
            if (frames == null) return -1;

            var best = -1;
            var bestDistance = PickTolerance;

            foreach (var frame in frames)
            {
                var distance = Mathf.Abs(frameToX(frame) - x);
                if (distance > bestDistance) continue;

                best = frame;
                bestDistance = distance;
            }

            return best;
        }

        /// <summary>Every key whose drawn position falls inside the horizontal band.</summary>
        public static void PickRange(IReadOnlyList<int> frames, float fromX, float toX,
            Func<int, float> frameToX, List<int> results)
        {
            results.Clear();
            if (frames == null) return;

            var left = Mathf.Min(fromX, toX) - PickTolerance;
            var right = Mathf.Max(fromX, toX) + PickTolerance;

            foreach (var frame in frames)
            {
                var x = frameToX(frame);
                if (x >= left && x <= right) results.Add(frame);
            }
        }
    }
}
