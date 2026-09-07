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
    /// <see cref="AnimationTagging.Normalise"/> can cancel keys against each other and the caller
    /// has to be able to compare what it asked for against what it got.
    /// <para>
    /// An edit clamps only the frames it produces. Keys already outside the clip - left there by a
    /// trim - are none of an unrelated edit's business, and deleting them is what this whole frame
    /// space was changed to stop.
    /// </para>
    /// </remarks>
    public static class AnimationTagEdits
    {
        /// <summary>
        /// The span of frames these channels hold keys on, for framing the view. False when there is
        /// nothing keyed anywhere.
        /// </summary>
        public static bool ContentRange(IReadOnlyList<AnimationTagging.TagChannel> channels,
            out int firstFrame, out int lastFrame)
        {
            firstFrame = int.MaxValue;
            lastFrame = int.MinValue;

            if (channels != null)
            {
                foreach (var channel in channels)
                {
                    if (channel?.toggles == null) continue;

                    foreach (var frame in channel.toggles)
                    {
                        firstFrame = Mathf.Min(firstFrame, frame);
                        lastFrame = Mathf.Max(lastFrame, frame);
                    }
                }
            }

            if (firstFrame <= lastFrame) return true;

            firstFrame = 0;
            lastFrame = 0;
            return false;
        }

        public static List<int> Insert(IReadOnlyList<int> toggles, int frame, int clipFrameCount) =>
            Normalised(toggles, InRange(frame, clipFrameCount));

        public static List<int> Delete(IReadOnlyList<int> toggles, IReadOnlyList<int> frames)
        {
            var removed = new HashSet<int>(frames);
            var result = new List<int>(toggles.Count);

            foreach (var toggle in toggles)
            {
                if (!removed.Contains(toggle)) result.Add(toggle);
            }

            AnimationTagging.Normalise(result);
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
            int clipFrameCount)
        {
            var moved = new HashSet<int>(frames);
            var result = new List<int>(toggles.Count);

            // Only a key this edit actually moves is clamped. A key already outside the clip - left
            // there by a trim - passes through untouched, because an unrelated edit must not tidy it
            // away.
            foreach (var toggle in toggles)
            {
                result.Add(moved.Contains(toggle) ? InRange(toggle + delta, clipFrameCount) : toggle);
            }

            AnimationTagging.Normalise(result);
            return result;
        }

        /// <summary>Adds a copy of each selected key at <paramref name="delta"/> frames away.</summary>
        public static List<int> Duplicate(IReadOnlyList<int> toggles, IReadOnlyList<int> frames,
            int delta, int clipFrameCount)
        {
            var result = new List<int>(toggles);
            foreach (var frame in frames) result.Add(InRange(frame + delta, clipFrameCount));

            AnimationTagging.Normalise(result);
            return result;
        }

        /// <summary>Scales the selected keys' distance from <paramref name="pivot"/>.</summary>
        public static List<int> Scale(IReadOnlyList<int> toggles, IReadOnlyList<int> frames,
            int pivot, float factor, int clipFrameCount)
        {
            var scaled = new HashSet<int>(frames);
            var result = new List<int>(toggles.Count);

            foreach (var toggle in toggles)
            {
                result.Add(scaled.Contains(toggle)
                    ? InRange(pivot + Mathf.RoundToInt((toggle - pivot) * factor), clipFrameCount)
                    : toggle);
            }

            AnimationTagging.Normalise(result);
            return result;
        }

        /// <summary>
        /// The largest part of <paramref name="delta"/> that keeps every selected key inside the
        /// clip, so a group drag stops at the edge instead of quietly deleting the keys that ran off.
        /// </summary>
        public static int ClampDelta(IReadOnlyList<int> frames, int delta, int clipFrameCount)
        {
            if (frames == null || frames.Count == 0) return 0;

            var lowest = int.MaxValue;
            var highest = int.MinValue;
            foreach (var frame in frames)
            {
                lowest = Mathf.Min(lowest, frame);
                highest = Mathf.Max(highest, frame);
            }

            return Mathf.Clamp(delta, -lowest, Mathf.Max(0, clipFrameCount - highest));
        }

        /// <summary>The frames the selection would land on, for previewing a move before it happens.</summary>
        public static void Preview(IReadOnlyList<int> frames, int delta, List<int> results)
        {
            results.Clear();
            foreach (var frame in frames) results.Add(frame + delta);
        }

        /// <summary>A frame an edit produced, kept inside the clip it belongs to.</summary>
        private static int InRange(int frame, int clipFrameCount) =>
            Mathf.Clamp(frame, 0, Mathf.Max(0, clipFrameCount));

        private static List<int> Normalised(IReadOnlyList<int> toggles, params int[] extra)
        {
            var result = new List<int>(toggles);
            result.AddRange(extra);

            AnimationTagging.Normalise(result);
            return result;
        }
    }
}
