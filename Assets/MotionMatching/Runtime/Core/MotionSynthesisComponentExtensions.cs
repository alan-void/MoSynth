using AnimationTools;

namespace MotionMatching
{
    public static class MotionSynthesisComponentExtensions
    {
        /// <summary>
        /// The <see cref="MotionMatchingData"/> of the first stage in
        /// <see cref="MotionSynthesisComponent.stages"/> that has one, or null when no such stage
        /// exists (e.g. a motion-field-only stack).
        /// </summary>
        /// <remarks>
        /// Matched on <see cref="IMotionMatchingDataProvider"/> rather than on
        /// <see cref="MotionMatchingStage"/>, because what every control input needs from here is
        /// the database's feature definitions and prediction horizons — not the search that reads
        /// them. A learned matcher answers this and reuses the control inputs unchanged.
        /// </remarks>
        public static MotionMatchingData GetMmData(this MotionSynthesisComponent msc)
        {
            foreach (var stage in msc.stages)
            {
                if (stage is IMotionMatchingDataProvider provider) return provider.MmData;
            }

            return null;
        }
    }
}
