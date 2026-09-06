using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The list arithmetic behind dragging, adding and deleting footfall anchors, kept apart from
    /// the drawing so the rules that lose data if they are wrong can be tested.
    /// </summary>
    /// <remarks>
    /// Two of those rules are silent when broken. <c>GaitPhase</c> drops any anchor that is not
    /// strictly later than the one before it, and <see cref="GaitPhaseComponent.OnValidate"/> drops
    /// any anchor outside the slice - neither reports anything. So a move clamps to the slice and
    /// returns an ascending list, always.
    /// </remarks>
    public static class FootfallEdits
    {
        /// <summary>
        /// The largest delta that keeps every selected anchor inside the slice, so a drag stops at
        /// the edge rather than pushing anchors off it to be deleted on commit.
        /// </summary>
        public static int ClampDelta(IReadOnlyList<GaitPhase.Footfall> footfalls,
            ICollection<int> selection, int delta, int clipFrameCount)
        {
            if (clipFrameCount <= 0 || selection.Count == 0) return 0;

            var lowest = int.MaxValue;
            var highest = int.MinValue;

            foreach (var index in selection)
            {
                if (index < 0 || index >= footfalls.Count) continue;

                lowest = Mathf.Min(lowest, footfalls[index].frame);
                highest = Mathf.Max(highest, footfalls[index].frame);
            }

            if (lowest == int.MaxValue) return 0;

            return Mathf.Clamp(delta, -lowest, clipFrameCount - 1 - highest);
        }

        /// <summary>
        /// Moves the selected anchors by a clamped delta and returns the list in ascending frame
        /// order. Never drops an entry: a collision keeps both, and the caller decides what to do
        /// about a repeated frame.
        /// </summary>
        public static List<GaitPhase.Footfall> Move(IReadOnlyList<GaitPhase.Footfall> footfalls,
            ICollection<int> selection, int delta, int clipFrameCount)
        {
            var clamped = ClampDelta(footfalls, selection, delta, clipFrameCount);
            var moved = new List<GaitPhase.Footfall>(footfalls.Count);

            for (var i = 0; i < footfalls.Count; i++)
            {
                var footfall = footfalls[i];
                if (selection.Contains(i))
                {
                    footfall.frame = Mathf.Clamp(footfall.frame + clamped, 0,
                        Mathf.Max(0, clipFrameCount - 1));
                }

                moved.Add(footfall);
            }

            Sort(moved);
            return moved;
        }

        /// <summary>
        /// Adds an anchor at a slice frame, choosing the foot opposite the nearest earlier anchor -
        /// the right guess when the reason you are adding one is a contact detection missed.
        /// </summary>
        public static List<GaitPhase.Footfall> Add(IReadOnlyList<GaitPhase.Footfall> footfalls,
            int clipFrame, int clipFrameCount)
        {
            var frame = Mathf.Clamp(clipFrame, 0, Mathf.Max(0, clipFrameCount - 1));
            var foot = GaitPhase.Foot.Left;

            for (var i = footfalls.Count - 1; i >= 0; i--)
            {
                if (footfalls[i].frame > frame) continue;

                foot = footfalls[i].foot == GaitPhase.Foot.Left
                    ? GaitPhase.Foot.Right
                    : GaitPhase.Foot.Left;
                break;
            }

            var result = new List<GaitPhase.Footfall>(footfalls) { new(frame, foot) };
            Sort(result);
            return result;
        }

        public static List<GaitPhase.Footfall> Delete(IReadOnlyList<GaitPhase.Footfall> footfalls,
            ICollection<int> selection)
        {
            var result = new List<GaitPhase.Footfall>(footfalls.Count);
            for (var i = 0; i < footfalls.Count; i++)
            {
                if (!selection.Contains(i)) result.Add(footfalls[i]);
            }

            return result;
        }

        /// <summary>Indices of anchors that repeat the previous anchor's foot - the missed contacts.</summary>
        public static List<int> RepeatedFeet(IReadOnlyList<GaitPhase.Footfall> footfalls)
        {
            var repeated = new List<int>();
            for (var i = 1; i < footfalls.Count; i++)
            {
                if (footfalls[i].foot == footfalls[i - 1].foot) repeated.Add(i);
            }

            return repeated;
        }

        private static void Sort(List<GaitPhase.Footfall> footfalls) =>
            footfalls.Sort((a, b) => a.frame.CompareTo(b.frame));
    }
}
