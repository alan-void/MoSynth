using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Rasterises a clip's gait phase and foot contacts into a strip, one pixel column per slice
    /// frame: the phase as a filled band, and each foot's contact as a bar under it.
    /// </summary>
    /// <remarks>
    /// Kept apart from the track that draws it because it is pure - given phase, rate and contact
    /// arrays it produces pixels and nothing else, so its two structural rules (a frame with no
    /// measurable cycle flags the whole band; a full cycle reaches the top row) are testable.
    /// </remarks>
    public static class GaitPhaseSignalBuilder
    {
        public const int PhaseRows = 34;
        public const int FlagRows = 4;
        public const int ContactRows = 6;
        public const int Height = PhaseRows + FlagRows + ContactRows * 2;

        public static readonly Color32 Background = new(36, 36, 38, 255);
        public static readonly Color32 PhaseFill = new(92, 148, 214, 255);
        public static readonly Color32 NoCycle = new(196, 72, 62, 255);
        public static readonly Color32 LeftContact = new(88, 176, 112, 255);
        public static readonly Color32 RightContact = new(206, 158, 74, 255);
        public static readonly Color32 MissedContact = new(214, 92, 196, 255);

        /// <summary>
        /// Fills <paramref name="pixels"/>, which must already be cleared to the background.
        /// <paramref name="columnToFrame"/> maps a column to the slice frame it stands for, so a
        /// clip longer than the texture is downsampled rather than truncated.
        /// </summary>
        public static void Fill(Color32[] pixels, int width, float[] phase, float[] phaseRate,
            bool[] contacts, System.Func<int, int> columnToFrame)
        {
            for (var x = 0; x < width; x++)
            {
                var frame = Mathf.Clamp(columnToFrame(x), 0, phase.Length - 1);

                DrawPhaseColumn(pixels, width, x, phase[frame], phaseRate[frame]);
                DrawContactColumn(pixels, width, x, frame, contacts);
            }
        }

        private static void DrawPhaseColumn(Color32[] pixels, int width, int x, float phaseValue, float rate)
        {
            // Rows are indexed from the texture's bottom, so the phase band sits at the top.
            var bandBottom = Height - PhaseRows;
            var filled = Mathf.Clamp(Mathf.RoundToInt(phaseValue / GaitPhase.Tau * (PhaseRows - 1)),
                0, PhaseRows - 1);

            // A frame with no cycle has no phase worth drawing, so the whole band flags instead.
            if (rate == 0f)
            {
                for (var row = 0; row < PhaseRows; row++)
                {
                    pixels[(bandBottom + row) * width + x] = Dim(NoCycle, row == filled ? 1f : 0.28f);
                }

                for (var row = 0; row < FlagRows; row++)
                {
                    pixels[(bandBottom - FlagRows + row) * width + x] = NoCycle;
                }

                return;
            }

            for (var row = 0; row <= filled; row++)
            {
                pixels[(bandBottom + row) * width + x] = Dim(PhaseFill, row == filled ? 1f : 0.45f);
            }
        }

        private static void DrawContactColumn(Color32[] pixels, int width, int x, int frame, bool[] contacts)
        {
            if (contacts == null || contacts.Length < (frame + 1) * 2) return;

            var rightBottom = 0;
            var leftBottom = ContactRows;

            if (contacts[frame * 2])
            {
                for (var row = 0; row < ContactRows; row++)
                {
                    pixels[(leftBottom + row) * width + x] = LeftContact;
                }
            }

            if (contacts[frame * 2 + 1])
            {
                for (var row = 0; row < ContactRows; row++)
                {
                    pixels[(rightBottom + row) * width + x] = RightContact;
                }
            }
        }

        private static Color32 Dim(Color32 colour, float scale) => new(
            (byte)(colour.r * scale), (byte)(colour.g * scale), (byte)(colour.b * scale), 255);

        /// <summary>The colour an anchor is drawn in; a repeated foot is set apart deliberately.</summary>
        public static Color AnchorColour(GaitPhase.Foot foot, bool repeated) => repeated
            ? (Color)MissedContact
            : foot == GaitPhase.Foot.Left ? (Color)LeftContact : (Color)RightContact;
    }
}
