using System;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// A dense per-frame signal rasterised once into a texture, one pixel column per frame, and
    /// drawn at any zoom as a single blit with a moving UV window.
    /// </summary>
    /// <remarks>
    /// This is what makes the timeline independent of clip length. A per-frame vector draw costs one
    /// primitive per frame every repaint; this costs one draw call, and zooming or panning changes
    /// only the UVs. The rebuild is the expensive part, so it happens only when the data version
    /// moves.
    /// </remarks>
    public sealed class TimelineSignalTexture : IDisposable
    {
        /// <summary>
        /// Well inside the platform maximum. A longer signal is downsampled to one frame per column,
        /// which is all a strip a few hundred pixels wide can show anyway.
        /// </summary>
        public const int MaxWidth = 4096;

        private Texture2D _texture;
        private int _frameCount = -1;
        private int _builtVersion = int.MinValue;

        /// <summary>Columns in the texture, which is the frame count unless it was downsampled.</summary>
        public int Width => _texture != null ? _texture.width : 0;

        /// <summary>The frame a column stands for, after any downsampling.</summary>
        public int ColumnToFrame(int column)
        {
            if (_frameCount <= 0) return 0;
            if (Width >= _frameCount) return Mathf.Min(column, _frameCount - 1);

            return Mathf.Min(_frameCount - 1,
                Mathf.RoundToInt((float)column / (Width - 1) * (_frameCount - 1)));
        }

        /// <summary>
        /// Rebuilds only when the signal's shape or version has moved. <paramref name="fill"/>
        /// receives a background-cleared buffer, its width, and its height, and is called with this
        /// instance's <see cref="ColumnToFrame"/> already valid.
        /// </summary>
        public void Rebuild(int frameCount, int height, int dataVersion, Color32 background,
            Action<Color32[], int, int> fill)
        {
            var width = Mathf.Clamp(frameCount, 1, MaxWidth);

            var shapeChanged = _texture == null || _texture.width != width || _texture.height != height;
            if (!shapeChanged && _builtVersion == dataVersion && _frameCount == frameCount) return;

            _frameCount = frameCount;
            _builtVersion = dataVersion;

            if (shapeChanged)
            {
                if (_texture != null) UnityEngine.Object.DestroyImmediate(_texture);
                _texture = new Texture2D(width, height, TextureFormat.RGBA32, false)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    hideFlags = HideFlags.HideAndDontSave
                };
            }

            var pixels = new Color32[width * height];
            for (var i = 0; i < pixels.Length; i++) pixels[i] = background;

            fill(pixels, width, height);

            _texture.SetPixels32(pixels);
            _texture.Apply(false);
        }

        /// <summary>Forces the next <see cref="Rebuild"/> to do the work regardless of version.</summary>
        public void Invalidate() => _builtVersion = int.MinValue;

        /// <summary>Blits the visible slice of the signal. Nothing is drawn when it is off screen.</summary>
        public void Draw(Rect destination, Rect uv)
        {
            if (_texture == null || destination.width <= 0f || destination.height <= 0f) return;

            GUI.DrawTextureWithTexCoords(destination, _texture, uv, true);
        }

        /// <summary>
        /// Maps a contiguous clip-frame range onto the part of a lane that is actually on screen,
        /// and the UV window of the signal that lands there.
        /// </summary>
        /// <returns>False when the range is entirely off screen, in which case nothing should draw.</returns>
        public static bool TryMapRange(in ClipTimeAxis axis, Rect laneRect, int firstFrame, int lastFrame,
            out Rect destination, out Rect uv)
        {
            destination = default;
            uv = default;

            var span = lastFrame - firstFrame;
            if (span <= 0) return false;

            var left = axis.FrameToX(firstFrame);
            var right = axis.FrameToX(lastFrame);
            if (right - left <= 0f) return false;

            var visibleLeft = Mathf.Max(left, laneRect.xMin);
            var visibleRight = Mathf.Min(right, laneRect.xMax);
            if (visibleRight - visibleLeft <= 0f) return false;

            destination = new Rect(visibleLeft, laneRect.y, visibleRight - visibleLeft, laneRect.height);

            var u0 = (visibleLeft - left) / (right - left);
            var u1 = (visibleRight - left) / (right - left);
            uv = new Rect(u0, 0f, u1 - u0, 1f);
            return true;
        }

        public void Dispose()
        {
            if (_texture != null) UnityEngine.Object.DestroyImmediate(_texture);
            _texture = null;
            _builtVersion = int.MinValue;
            _frameCount = -1;
        }
    }
}
