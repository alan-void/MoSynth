using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Picks footfall anchors by pixel position: which one is under the cursor, and which fall
    /// inside a rubber-band selection.
    /// </summary>
    /// <remarks>
    /// Pure and separate from the track so it can be tested, because the interesting cases are the
    /// awkward ones - two anchors landing on the same pixel when zoomed out, and a click just past
    /// the tolerance that must select nothing rather than the nearest thing anywhere.
    /// </remarks>
    public static class FootfallHitTester
    {
        /// <summary>How far from an anchor's centre a click still counts, in pixels.</summary>
        public const float PickTolerance = 6f;

        /// <summary>
        /// The anchor nearest <paramref name="x"/> within <see cref="PickTolerance"/>, or -1.
        /// Ties go to the earlier anchor, so picking is stable when two share a pixel.
        /// </summary>
        public static int Pick(IReadOnlyList<GaitPhase.Footfall> footfalls, float x,
            System.Func<int, float> clipFrameToX)
        {
            var best = -1;
            var bestDistance = PickTolerance;

            for (var i = 0; i < footfalls.Count; i++)
            {
                var distance = Mathf.Abs(clipFrameToX(footfalls[i].frame) - x);
                if (distance >= bestDistance) continue;

                bestDistance = distance;
                best = i;
            }

            return best;
        }

        /// <summary>Every anchor whose x lies within the horizontal span, inclusive.</summary>
        public static void PickRange(IReadOnlyList<GaitPhase.Footfall> footfalls, float minX, float maxX,
            System.Func<int, float> clipFrameToX, ICollection<int> results)
        {
            if (minX > maxX) (minX, maxX) = (maxX, minX);

            for (var i = 0; i < footfalls.Count; i++)
            {
                var x = clipFrameToX(footfalls[i].frame);
                if (x >= minX && x <= maxX) results.Add(i);
            }
        }
    }
}
