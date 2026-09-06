using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Picks the ruler's tick spacing for a zoom level, so labels stay readable from a whole clip
    /// down to a single frame without ever drawing more ticks than the ruler has pixels.
    /// </summary>
    public static class ClipTimelineTicks
    {
        /// <summary>Steps a reader can do arithmetic in; anything else makes a ruler harder to read.</summary>
        private static readonly int[] Ladder =
            { 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000 };

        private const float MinMajorSpacing = 60f;
        private const float MinMinorSpacing = 6f;

        /// <summary>
        /// The smallest ladder steps whose on-screen spacing stays legible. <paramref name="major"/>
        /// carries a label, <paramref name="minor"/> is an unlabelled subdivision of it and is 0
        /// when no subdivision fits.
        /// </summary>
        public static void Choose(float pixelsPerFrame, out int major, out int minor)
        {
            major = SmallestStepWithSpacing(pixelsPerFrame, MinMajorSpacing);
            minor = SmallestStepWithSpacing(pixelsPerFrame, MinMinorSpacing);

            // A minor that does not divide its major draws ticks that drift off the labelled ones.
            if (minor >= major || major % minor != 0) minor = 0;
        }

        private static int SmallestStepWithSpacing(float pixelsPerFrame, float minSpacing)
        {
            foreach (var step in Ladder)
            {
                if (step * pixelsPerFrame >= minSpacing) return step;
            }

            return Ladder[Ladder.Length - 1];
        }

        /// <summary>
        /// The first multiple of <paramref name="step"/> at or after <paramref name="fromFrame"/>,
        /// which is where a culled tick loop starts.
        /// </summary>
        public static int FirstTickAtOrAfter(int fromFrame, int step)
        {
            if (step <= 0) return fromFrame;

            var remainder = fromFrame % step;
            if (remainder == 0) return fromFrame;

            return fromFrame < 0 ? fromFrame - remainder : fromFrame + (step - remainder);
        }

        /// <summary>Formats a major tick's label as a frame number or as seconds.</summary>
        public static string Label(int frame, float frameTime, bool asSeconds)
        {
            if (!asSeconds) return frame.ToString();

            var seconds = frame * frameTime;
            return seconds < 10f ? $"{seconds:0.00}s" : $"{seconds:0.0}s";
        }
    }
}
