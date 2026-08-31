using System;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// A timeline of a clip's gait phase: the phase itself, the contacts it was derived from, the
    /// footfall anchors, and the two ways the phase is known to go wrong.
    /// </summary>
    /// <remarks>
    /// Drawn as a generated <see cref="Texture2D"/>, one pixel column per frame, rather than as
    /// vector line drawing. A clip runs to several thousand frames, and a texture costs one blit
    /// per repaint regardless of length while a per-frame line draw would not.
    /// <para>
    /// The point of showing this in the inspector rather than as an offline plot is the preview
    /// beside it: clicking a column poses the rig at that frame, so a suspicious span can be
    /// checked against the actual motion.
    /// </para>
    /// </remarks>
    public sealed class GaitPhaseStrip : IDisposable
    {
        private const int PhaseRows = 26;
        private const int FlagRows = 4;
        private const int ContactRows = 6;
        private const int FootfallRows = 6;
        private const int Height = PhaseRows + FlagRows + ContactRows * 2 + FootfallRows;

        // Texture width is capped well inside the platform maximum; a longer clip is downsampled by
        // taking one frame per column, which is what a strip this size can show anyway.
        private const int MaxWidth = 4096;

        private static readonly Color32 Background = new(36, 36, 38, 255);
        private static readonly Color32 PhaseFill = new(92, 148, 214, 255);
        private static readonly Color32 NoCycle = new(196, 72, 62, 255);
        private static readonly Color32 LeftContact = new(88, 176, 112, 255);
        private static readonly Color32 RightContact = new(206, 158, 74, 255);
        private static readonly Color32 MissedContact = new(214, 92, 196, 255);

        private Texture2D _texture;
        private bool[] _contacts = Array.Empty<bool>();
        private bool _dirty = true;

        /// <summary>Rebuild on the next draw. Cheap; the rebuild itself is the cost.</summary>
        public void Invalidate() => _dirty = true;

        /// <summary>
        /// The contact flags a detection pass produced, so the strip can show what the footfalls
        /// were read from rather than only the footfalls themselves.
        /// </summary>
        public void SetContacts(bool[] contacts)
        {
            _contacts = contacts ?? Array.Empty<bool>();
            _dirty = true;
        }

        /// <summary>
        /// Draws the strip and its summary. <paramref name="seekToClipFrame"/> receives a
        /// <em>whole-clip</em> frame, since that is what the preview scrubs.
        /// </summary>
        public void Draw(AnnotatedAnimationClip clip, GaitPhaseComponent phase,
            Action<int> seekToClipFrame)
        {
            var frameCount = clip.FrameCount;
            if (frameCount <= 0)
            {
                EditorGUILayout.HelpBox("The clip's frame range is empty.", MessageType.Info);
                return;
            }

            phase.Evaluate(clip, out var phaseValues, out var phaseRates);

            if (_dirty || _texture == null || _texture.width != Mathf.Min(frameCount, MaxWidth))
            {
                Rebuild(phase, phaseValues, phaseRates, frameCount);
                _dirty = false;
            }

            var rect = GUILayoutUtility.GetRect(10f, Height, GUILayout.ExpandWidth(true));
            GUI.DrawTexture(rect, _texture, ScaleMode.StretchToFill);

            HandleClick(rect, clip, frameCount, seekToClipFrame);
            DrawSummary(phase, phaseValues, phaseRates, frameCount, clip.FrameTime);
        }

        private void HandleClick(Rect rect, AnnotatedAnimationClip clip, int frameCount,
            Action<int> seekToClipFrame)
        {
            var e = Event.current;
            if (e.type != EventType.MouseDown && e.type != EventType.MouseDrag) return;
            if (!rect.Contains(e.mousePosition)) return;

            var fraction = Mathf.Clamp01((e.mousePosition.x - rect.x) / rect.width);
            var sliceFrame = Mathf.Clamp(Mathf.RoundToInt(fraction * (frameCount - 1)), 0, frameCount - 1);

            // The strip is in sliced frames; the preview is not.
            seekToClipFrame(clip.startFrame + sliceFrame);
            e.Use();
        }

        private void DrawSummary(GaitPhaseComponent phase, float[] phaseValues, float[] phaseRates,
            int frameCount, float frameTime)
        {
            var noCycle = 0;
            foreach (var rate in phaseRates)
            {
                if (rate == 0f) noCycle++;
            }

            var repeats = GaitPhase.CountRepeatedFeet(phase.footfalls);
            var left = 0;
            foreach (var footfall in phase.footfalls)
            {
                if (footfall.foot == GaitPhase.Foot.Left) left++;
            }

            EditorGUILayout.LabelField(
                $"{phase.footfalls.Count} footfalls ({left} L / {phase.footfalls.Count - left} R)   " +
                $"stride {StrideSeconds(phaseRates, frameTime):0.00}s   " +
                $"no cycle on {100f * noCycle / frameCount:0.#}% of {frameCount} frames",
                EditorStyles.miniLabel);

            if (repeats > 0)
            {
                EditorGUILayout.HelpBox(
                    $"{repeats} footfalls are the same foot twice running. Each one is a contact the " +
                    "detection missed, and the phase jumps a whole cycle there instead of half. Raise " +
                    "the velocity threshold, or add the missing footfall by hand.",
                    MessageType.Warning);
            }

            if (noCycle > 0)
            {
                EditorGUILayout.HelpBox(
                    $"{noCycle} frames sit outside the first and last footfall and have no measurable " +
                    "cycle. Training drops them. Trimming the clip to the walking part is usually the " +
                    "right fix.",
                    MessageType.Info);
            }
        }

        /// <summary>Mean stride period over the frames that have a cycle, in seconds.</summary>
        private static float StrideSeconds(float[] phaseRates, float frameTime)
        {
            var total = 0f;
            var counted = 0;
            foreach (var rate in phaseRates)
            {
                if (rate <= 0f) continue;
                total += rate;
                counted++;
            }

            return counted == 0 || total <= 0f ? 0f : GaitPhase.Tau / (total / counted);
        }

        private void Rebuild(GaitPhaseComponent phase, float[] phaseValues, float[] phaseRates,
            int frameCount)
        {
            var width = Mathf.Min(frameCount, MaxWidth);

            if (_texture == null || _texture.width != width)
            {
                if (_texture != null) UnityEngine.Object.DestroyImmediate(_texture);
                _texture = new Texture2D(width, Height, TextureFormat.RGBA32, false)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    hideFlags = HideFlags.HideAndDontSave
                };
            }

            var pixels = new Color32[width * Height];
            for (var i = 0; i < pixels.Length; i++) pixels[i] = Background;

            for (var x = 0; x < width; x++)
            {
                var frame = width == frameCount
                    ? x
                    : Mathf.Min(frameCount - 1, Mathf.RoundToInt((float)x / (width - 1) * (frameCount - 1)));

                DrawPhaseColumn(pixels, width, x, phaseValues[frame], phaseRates[frame]);
                DrawContactColumn(pixels, width, x, frame);
            }

            DrawFootfalls(pixels, width, phase, frameCount);

            _texture.SetPixels32(pixels);
            _texture.Apply(false);
        }

        private static void DrawPhaseColumn(Color32[] pixels, int width, int x, float phaseValue,
            float rate)
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

        private void DrawContactColumn(Color32[] pixels, int width, int x, int frame)
        {
            if (_contacts.Length < (frame + 1) * 2) return;

            var rightBottom = FootfallRows;
            var leftBottom = FootfallRows + ContactRows;

            if (_contacts[frame * 2])
            {
                for (var row = 0; row < ContactRows; row++)
                {
                    pixels[(leftBottom + row) * width + x] = LeftContact;
                }
            }

            if (_contacts[frame * 2 + 1])
            {
                for (var row = 0; row < ContactRows; row++)
                {
                    pixels[(rightBottom + row) * width + x] = RightContact;
                }
            }
        }

        /// <summary>
        /// One tick per anchor, drawn last so a tick is never hidden by a contact bar. A repeated
        /// foot is coloured apart, because that is where the phase gains a whole cycle.
        /// </summary>
        private static void DrawFootfalls(Color32[] pixels, int width, GaitPhaseComponent phase,
            int frameCount)
        {
            for (var i = 0; i < phase.footfalls.Count; i++)
            {
                var footfall = phase.footfalls[i];
                if (footfall.frame < 0 || footfall.frame >= frameCount) continue;

                var x = frameCount <= 1
                    ? 0
                    : Mathf.Clamp(Mathf.RoundToInt((float)footfall.frame / (frameCount - 1) * (width - 1)),
                        0, width - 1);

                var repeated = i > 0 && phase.footfalls[i - 1].foot == footfall.foot;
                var colour = repeated
                    ? MissedContact
                    : footfall.foot == GaitPhase.Foot.Left ? LeftContact : RightContact;

                for (var row = 0; row < FootfallRows; row++)
                {
                    pixels[row * width + x] = colour;
                }
            }
        }

        private static Color32 Dim(Color32 colour, float scale) => new(
            (byte)(colour.r * scale), (byte)(colour.g * scale), (byte)(colour.b * scale), 255);

        public void Dispose()
        {
            if (_texture != null) UnityEngine.Object.DestroyImmediate(_texture);
            _texture = null;
        }
    }
}
