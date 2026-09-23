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
        /// <remarks>See <see cref="IMotionMatchingDataProvider"/> for why this is not matched on the stage type.</remarks>
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
