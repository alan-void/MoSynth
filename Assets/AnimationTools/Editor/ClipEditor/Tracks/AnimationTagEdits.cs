using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The arithmetic behind every edit to a tag channel's keys, kept apart from the drawing so it
    /// can be tested against the cases that lose data.
    /// </summary>
    /// <remarks>
    /// Every method returns a new, normalised list rather than mutating in place, because
    /// <see cref="AnimationTagging.Normalise"/> can drop or cancel keys and the caller has to be
    /// able to compare what it asked for against what it got.
    /// </remarks>
    public static class AnimationTagEdits
    {
        public static List<int> Insert(IReadOnlyList<int> toggles, int frame, int frameCount) =>
            Normalised(toggles, frameCount, frame);

        public static List<int> Delete(IReadOnlyList<int> toggles, IReadOnlyList<int> frames,
            int frameCount)
        {
            var removed = new HashSet<int>(frames);
            var result = new List<int>(toggles.Count);

            foreach (var toggle in toggles)
            {
                if (!removed.Contains(toggle)) result.Add(toggle);
            }

            AnimationTagging.Normalise(result, frameCount);
            return result;
        }

        /// <summary>
        /// Moves the selected keys by <paramref name="delta"/>, leaving the rest where they are.
        /// </summary>
        /// <remarks>
        /// A moved key landing on a stationary one is not an error: the two cancel in
        /// <see cref="AnimationTagging.Normalise"/>, which is the same flip-and-flip-back the signal
        /// already meant. Callers re-derive their selection from the returned list rather than
        /// assuming every moved key survived.
        /// </remarks>
        public static List<int> Move(IReadOnlyList<int> toggles, IReadOnlyList<int> frames, int delta,
            int frameCount)
        {
            var moved = new HashSet<int>(frames);
            var result = new List<int>(toggles.Count);

            foreach (var toggle in toggles)
            {
                result.Add(moved.Contains(toggle) ? toggle + delta : toggle);
            }

            AnimationTagging.Normalise(result, frameCount);
            return result;
        }

        /// <summary>Adds a copy of each selected key at <paramref name="delta"/> frames away.</summary>
        public static List<int> Duplicate(IReadOnlyList<int> toggles, IReadOnlyList<int> frames,
            int delta, int frameCount)
        {
            var result = new List<int>(toggles);
            foreach (var frame in frames) result.Add(frame + delta);

            AnimationTagging.Normalise(result, frameCount);
            return result;
        }

        /// <summary>Scales the selected keys' distance from <paramref name="pivot"/>.</summary>
        public static List<int> Scale(IReadOnlyList<int> toggles, IReadOnlyList<int> frames,
            int pivot, float factor, int frameCount)
        {
            var scaled = new HashSet<int>(frames);
            var result = new List<int>(toggles.Count);

            foreach (var toggle in toggles)
            {
                result.Add(scaled.Contains(toggle)
                    ? pivot + Mathf.RoundToInt((toggle - pivot) * factor)
                    : toggle);
            }

            AnimationTagging.Normalise(result, frameCount);
            return result;
        }

        /// <summary>
        /// The largest part of <paramref name="delta"/> that keeps every selected key inside the
        /// clip, so a group drag stops at the edge instead of quietly deleting the keys that ran off.
        /// </summary>
        public static int ClampDelta(IReadOnlyList<int> frames, int delta, int frameCount)
        {
            if (frames == null || frames.Count == 0) return 0;

            var lowest = int.MaxValue;
            var highest = int.MinValue;
            foreach (var frame in frames)
            {
                lowest = Mathf.Min(lowest, frame);
                highest = Mathf.Max(highest, frame);
            }

            return Mathf.Clamp(delta, -lowest, Mathf.Max(0, frameCount - highest));
        }

        /// <summary>The frames the selection would land on, for previewing a move before it happens.</summary>
        public static void Preview(IReadOnlyList<int> frames, int delta, List<int> results)
        {
            results.Clear();
            foreach (var frame in frames) results.Add(frame + delta);
        }

        private static List<int> Normalised(IReadOnlyList<int> toggles, int frameCount,
            params int[] extra)
        {
            var result = new List<int>(toggles);
            result.AddRange(extra);

            AnimationTagging.Normalise(result, frameCount);
            return result;
        }
    }
}
