using System;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The one mapping between clip frames and timeline pixels: zoom, horizontal scroll, and the
    /// visible frame range everything else culls against.
    /// </summary>
    /// <remarks>
    /// A single instance is shared by the ruler, the slice overlay, the playhead and every track
    /// lane, so nothing else in the timeline may derive an x from a frame. Horizontal scrolling is
    /// this struct rather than a <c>GUI.BeginScrollView</c> for the same reason - two sources of
    /// horizontal offset is how a lane drifts out of step with the ruler above it.
    /// </remarks>
    [Serializable]
    public struct ClipTimeAxis
    {
        /// <summary>Zoomed all the way in, one frame is this wide.</summary>
        public const float MaxPixelsPerFrame = 40f;

        /// <summary>
        /// Zoomed all the way out. Small enough that even a long clip shrinks to under a pixel, so
        /// it is a floor on the arithmetic rather than on what you can look at.
        /// </summary>
        public const float MinPixelsPerFrame = 1e-4f;

        /// <summary>
        /// Below this a rect is a layout placeholder, not a measured pane, and is ignored.
        /// </summary>
        /// <remarks>
        /// <c>GUILayoutUtility.GetRect</c> hands back a dummy rect during a <c>Layout</c> pass.
        /// Seeding the zoom from that once drew a whole clip into a single pixel, and the fix used
        /// to be re-clamping to fit every frame - which is exactly the limit that made zooming out
        /// past the clip impossible. Ignoring the placeholder outright protects against the same bug
        /// without constraining the zoom.
        /// </remarks>
        private const float MinimumMeasuredWidth = 8f;

        public float pixelsPerFrame;

        /// <summary>The fractional frame sitting at <see cref="ViewRect"/>'s left edge.</summary>
        public float scrollFrames;

        /// <summary>Recomputed from the layout every repaint, so it is never serialized.</summary>
        [NonSerialized] public Rect ViewRect;

        /// <summary>
        /// Adopts this repaint's rect and seeds the zoom on first use. Call once per repaint before
        /// anything reads the axis.
        /// </summary>
        /// <remarks>
        /// A placeholder rect is ignored rather than adopted, so the zoom can never be seeded from a
        /// width the pane never had. That is what allows the zoom floor to be an absolute one:
        /// zooming out past the end of the clip is a view you are allowed to hold, and <c>Home</c>
        /// always brings the whole clip back.
        /// </remarks>
        public void Prepare(Rect viewRect, int frameCount)
        {
            if (viewRect.width <= MinimumMeasuredWidth) return;

            ViewRect = viewRect;

            pixelsPerFrame = pixelsPerFrame <= 0f
                ? FitPixelsPerFrame(frameCount)
                : Mathf.Clamp(pixelsPerFrame, MinPixelsPerFrame, MaxPixelsPerFrameFor(frameCount));
        }

        public float FrameToX(float frame) =>
            ViewRect.xMin + (frame - scrollFrames) * pixelsPerFrame;

        public float XToFrame(float x) =>
            pixelsPerFrame <= 0f ? 0f : scrollFrames + (x - ViewRect.xMin) / pixelsPerFrame;

        /// <summary>The fractional frame at the left edge, which may sit outside the clip.</summary>
        /// <remarks>
        /// The scroll is deliberately unbounded - the clip is content to look at, not a wall to bump
        /// into - so these two are the honest answer to "what is on screen" and
        /// <see cref="FirstVisibleFrame"/> is the answer clamped to frames that actually exist. Draw
        /// the ruler from these; index anything with the other pair.
        /// </remarks>
        public float LeftEdgeFrame => scrollFrames;

        /// <summary>The fractional frame at the right edge, which may sit outside the clip.</summary>
        public float RightEdgeFrame => scrollFrames + VisibleFrameSpan;

        /// <summary>How many frames the view spans, fractional.</summary>
        public float VisibleFrameSpan =>
            pixelsPerFrame <= 0f ? 0f : ViewRect.width / pixelsPerFrame;

        /// <summary>The zoom at which the whole clip exactly fills the view.</summary>
        public float FitPixelsPerFrame(int frameCount)
        {
            if (ViewRect.width <= 0f) return 1f;
            return ViewRect.width / Mathf.Max(1, frameCount);
        }

        /// <summary>
        /// A clip short enough that fitting it already exceeds <see cref="MaxPixelsPerFrame"/> still
        /// has to be showable whole, so the ceiling rises to meet the fit.
        /// </summary>
        private float MaxPixelsPerFrameFor(int frameCount) =>
            Mathf.Max(MaxPixelsPerFrame, FitPixelsPerFrame(frameCount));

        public int FirstVisibleFrame(int frameCount) =>
            Mathf.Clamp(Mathf.FloorToInt(scrollFrames), 0, Mathf.Max(0, frameCount - 1));

        public int LastVisibleFrame(int frameCount) =>
            Mathf.Clamp(Mathf.CeilToInt(scrollFrames + VisibleFrameSpan), 0, Mathf.Max(0, frameCount - 1));

        /// <summary>
        /// Scales the zoom about <paramref name="pivotX"/>, keeping whichever frame is under that
        /// pixel exactly where it is.
        /// </summary>
        public void ZoomAt(float pivotX, float factor, int frameCount)
        {
            var pivotFrame = XToFrame(pivotX);

            pixelsPerFrame = Mathf.Clamp(pixelsPerFrame * factor,
                MinPixelsPerFrame, MaxPixelsPerFrameFor(frameCount));

            scrollFrames = pivotFrame - (pivotX - ViewRect.xMin) / pixelsPerFrame;
        }

        /// <summary>Scrolls by a pixel delta in view space; positive moves the content left.</summary>
        /// <remarks>
        /// Nothing stops this at the ends of the clip. A pan that runs out of content is a view you
        /// are allowed to hold, the ruler keeps numbering into it, and <c>Home</c> is the way back.
        /// </remarks>
        public void PanPixels(float deltaX)
        {
            if (pixelsPerFrame <= 0f) return;

            scrollFrames += deltaX / pixelsPerFrame;
        }

        /// <summary>Zooms and scrolls so the inclusive frame range fills the view.</summary>
        public void FrameRange(int firstFrame, int lastFrame, int frameCount)
        {
            var span = Mathf.Max(1, lastFrame - firstFrame + 1);

            pixelsPerFrame = Mathf.Clamp(ViewRect.width <= 0f ? 1f : ViewRect.width / span,
                MinPixelsPerFrame, MaxPixelsPerFrameFor(frameCount));

            scrollFrames = firstFrame;
        }
    }
}
