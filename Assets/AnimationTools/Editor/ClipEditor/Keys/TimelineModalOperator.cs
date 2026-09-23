using System.Globalization;
using UnityEngine;

namespace AnimationTools.Editor
{
    public enum TimelineModalKind
    {
        None,
        Grab,
        Scale,
        BoxSelect
    }

    public enum TimelineModalResult
    {
        None,
        Started,
        Updated,
        Confirmed,
        Cancelled
    }

    /// <summary>
    /// A Blender-style modal operator: press a key, move the mouse with no button held, then
    /// confirm or cancel.
    /// </summary>
    /// <remarks>
    /// A track previews the value while the mode runs and commits once on
    /// <see cref="TimelineModalResult.Confirmed"/>, so a cancel writes nothing. The cursor is also
    /// sampled on <see cref="EventType.Repaint"/>, because in a UI Toolkit window <c>MouseMove</c>
    /// with no button held may never arrive. <see cref="ClipEditorContext"/> ensures only one runs.
    /// </remarks>
    public sealed class TimelineModalOperator
    {
        private int _controlId;
        private Vector2 _startMouse;
        private Vector2 _mouse;
        private string _typed = string.Empty;
        private float _pixelsPerFrame = 1f;
        private int _pivotFrame;
        private float _pivotX;

        /// <summary>
        /// Which mode this is, and it stays readable after the mode ends so the owner can act on
        /// what was confirmed.
        /// </summary>
        /// <remarks>
        /// Separate from <see cref="IsActive"/>: finishing must not clear it, since the caller
        /// switches on it to apply the confirmed edit.
        /// </remarks>
        public TimelineModalKind Kind { get; private set; }

        public bool IsActive { get; private set; }

        /// <summary>Frames to move the selection by. Zero for any other kind.</summary>
        public int FrameDelta { get; private set; }

        /// <summary>Factor to scale the selection about <see cref="PivotFrame"/> by.</summary>
        public float ScaleFactor { get; private set; } = 1f;

        public int PivotFrame => _pivotFrame;

        /// <summary>The box being dragged, in GUI space. Meaningful only while box selecting.</summary>
        public Rect BoxRect => Rect.MinMaxRect(
            Mathf.Min(_startMouse.x, _mouse.x), Mathf.Min(_startMouse.y, _mouse.y),
            Mathf.Max(_startMouse.x, _mouse.x), Mathf.Max(_startMouse.y, _mouse.y));

        /// <summary>Whether the operator extends the existing selection rather than replacing it.</summary>
        public bool Extend { get; private set; }

        /// <summary>The header hint, matching Blender's - "Move X: 14", "Resize: 1.25".</summary>
        public string StatusText
        {
            get
            {
                var typed = _typed.Length > 0 ? _typed : null;
                return Kind switch
                {
                    TimelineModalKind.Grab => $"Move X: {typed ?? FrameDelta.ToString()}",
                    TimelineModalKind.Scale =>
                        $"Resize: {typed ?? ScaleFactor.ToString("0.###", CultureInfo.InvariantCulture)}",
                    _ => string.Empty
                };
            }
        }

        /// <summary>
        /// Enters a mode. <paramref name="pixelsPerFrame"/> converts mouse travel to frames, and a
        /// scale happens about <paramref name="pivotFrame"/> - the playhead, as in Blender.
        /// </summary>
        /// <param name="controlId">
        /// Allocated by the caller with <c>GUIUtility.GetControlID</c> unconditionally every event,
        /// so IMGUI control ids do not drift between Layout and Repaint.
        /// </param>
        public void Begin(TimelineModalKind kind, int controlId, Vector2 mouse, float pixelsPerFrame,
            int pivotFrame, float pivotX, bool extend = false)
        {
            Kind = kind;
            IsActive = true;
            _controlId = controlId;
            _startMouse = mouse;
            _mouse = mouse;
            _typed = string.Empty;
            _pixelsPerFrame = Mathf.Max(0.0001f, pixelsPerFrame);
            _pivotFrame = pivotFrame;
            _pivotX = pivotX;
            Extend = extend;

            FrameDelta = 0;
            ScaleFactor = 1f;

            // Tests drive the operator with no GUI in progress to own a hot control.
            if (Event.current == null) return;

            GUIUtility.hotControl = controlId;
        }

        /// <summary>
        /// Ends the mode without confirming, and clears the kind a finished mode left readable.
        /// Callers must discard any preview.
        /// </summary>
        public void Cancel()
        {
            if (GUIUtility.hotControl == _controlId) GUIUtility.hotControl = 0;

            Kind = TimelineModalKind.None;
            IsActive = false;
            FrameDelta = 0;
            ScaleFactor = 1f;
            _typed = string.Empty;
        }

        /// <summary>
        /// Feeds one IMGUI event to the running mode. Consumes every event it sees, so nothing
        /// underneath - the timeline's own scrubbing included - reacts while a mode is up.
        /// </summary>
        public TimelineModalResult HandleEvent(Event e)
        {
            if (!IsActive || e == null) return TimelineModalResult.None;

            switch (e.type)
            {
                // Never Use() a repaint - it is a draw pass, not an input event. Sampling it is what
                // lets a mode follow the cursor in a UI Toolkit window, where MouseMove may not come.
                case EventType.Repaint:
                    if (_mouse == e.mousePosition) return TimelineModalResult.None;

                    _mouse = e.mousePosition;
                    Recompute();
                    return TimelineModalResult.Updated;

                case EventType.MouseMove:
                case EventType.MouseDrag:
                    _mouse = e.mousePosition;
                    Recompute();
                    e.Use();
                    return TimelineModalResult.Updated;

                case EventType.MouseDown when e.button == 0:
                case EventType.MouseUp when e.button == 0 && Kind == TimelineModalKind.BoxSelect:
                    _mouse = e.mousePosition;
                    Recompute();
                    e.Use();
                    return Finish(TimelineModalResult.Confirmed);

                case EventType.MouseDown when e.button == 1:
                    e.Use();
                    return Finish(TimelineModalResult.Cancelled);

                case EventType.KeyDown when e.keyCode is KeyCode.Return or KeyCode.KeypadEnter:
                    e.Use();
                    return Finish(TimelineModalResult.Confirmed);

                case EventType.KeyDown when e.keyCode == KeyCode.Escape:
                    e.Use();
                    return Finish(TimelineModalResult.Cancelled);

                case EventType.KeyDown:
                    e.Use();
                    return HandleTyping(e) ? TimelineModalResult.Updated : TimelineModalResult.None;

                // Swallowed so a mode is never half-dismissed by an event nobody handled.
                case EventType.MouseUp:
                case EventType.ScrollWheel:
                case EventType.KeyUp:
                    e.Use();
                    return TimelineModalResult.None;
            }

            return TimelineModalResult.None;
        }

        /// <summary>Typing a number overrides the mouse, exactly as it does in Blender.</summary>
        private bool HandleTyping(Event e)
        {
            if (e.keyCode == KeyCode.Backspace)
            {
                if (_typed.Length == 0) return false;

                _typed = _typed[..^1];
                Recompute();
                return true;
            }

            var c = e.character;
            var accepted = char.IsDigit(c)
                           || (c == '-' && _typed.Length == 0)
                           || (c == '.' && Kind == TimelineModalKind.Scale && !_typed.Contains('.'));

            if (!accepted) return false;

            _typed += c;
            Recompute();
            return true;
        }

        private void Recompute()
        {
            switch (Kind)
            {
                case TimelineModalKind.Grab:
                    FrameDelta = TryParseTyped(out var typedFrames)
                        ? Mathf.RoundToInt(typedFrames)
                        : Mathf.RoundToInt((_mouse.x - _startMouse.x) / _pixelsPerFrame);
                    return;

                case TimelineModalKind.Scale:
                    ScaleFactor = TryParseTyped(out var typedFactor)
                        ? typedFactor
                        : ScaleFromMouse();
                    return;
            }
        }

        /// <summary>
        /// The factor is the ratio of the cursor's distance from the pivot to where it started, so
        /// dragging away from the playhead spreads the keys and dragging towards it pulls them in.
        /// </summary>
        /// <remarks>The reference is floored at a pixel so a start on the pivot cannot divide by zero.</remarks>
        private float ScaleFromMouse()
        {
            var reference = _startMouse.x - _pivotX;
            if (Mathf.Abs(reference) < 1f) reference = reference < 0f ? -1f : 1f;

            return (_mouse.x - _pivotX) / reference;
        }

        private bool TryParseTyped(out float value)
        {
            value = 0f;
            if (_typed.Length == 0 || _typed == "-" || _typed == ".") return false;

            return float.TryParse(_typed, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        /// <summary>
        /// Stops the mode but leaves <see cref="Kind"/> and the value intact, because the caller
        /// reads both to apply the edit. <see cref="Cancel"/> is what clears them.
        /// </summary>
        private TimelineModalResult Finish(TimelineModalResult result)
        {
            if (GUIUtility.hotControl == _controlId) GUIUtility.hotControl = 0;

            if (result == TimelineModalResult.Cancelled)
            {
                FrameDelta = 0;
                ScaleFactor = 1f;
            }

            IsActive = false;
            _typed = string.Empty;
            return result;
        }
    }
}
