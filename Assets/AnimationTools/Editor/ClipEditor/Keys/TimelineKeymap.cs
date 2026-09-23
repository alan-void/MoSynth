using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>What a keystroke means in a keyframe lane.</summary>
    public enum TimelineKeyAction
    {
        None,
        SelectAll,
        DeselectAll,
        InvertSelection,
        BeginBoxSelect,
        BeginGrab,
        BeginScale,
        Duplicate,
        Delete,
        InsertKey,
        FrameAll,
        FrameSelected,
        StepBack,
        StepForward,
        PreviousKey,
        NextKey
    }

    /// <summary>
    /// The one place a key becomes an action, so every keyframe lane in the window agrees and the
    /// bindings can be read without opening two switch statements.
    /// </summary>
    /// <remarks>
    /// Modelled on Blender's dope sheet: <c>A</c> is select-all, so view framing sits on <c>Home</c>
    /// and <c>.</c>. <c>F</c> (frame the slice) belongs to <see cref="ClipTimelineView"/>, and
    /// <c>Space</c> is a shortcut-manager binding on <see cref="AnnotatedClipEditorWindow"/> so it
    /// works whichever pane has focus.
    /// </remarks>
    public static class TimelineKeymap
    {
        /// <summary>The lines shown in a track's inspector, since none of this is discoverable.</summary>
        public static readonly string[] Summary =
        {
            "click / shift-click   select / extend",
            "drag empty, or B      box select",
            "A / Alt+A / Ctrl+I    select all / none / invert",
            "G                     move selected      (type a number, Enter, or Esc)",
            "S                     scale about playhead",
            "Shift+D               duplicate",
            "X / Delete            delete selected",
            "I                     insert key at playhead",
            "Home / Shift+Home     frame the keys / the whole clip",
            ".                     frame selected",
            "Up / Down             jump to previous / next key",
            "Left / Right          step one frame",
            "Space                 play / pause"
        };

        public static TimelineKeyAction Resolve(Event e)
        {
            if (e == null || e.type != EventType.KeyDown) return TimelineKeyAction.None;

            // Checked before the bare keys, or Alt+A reads as A and Shift+D as D.
            if (e.alt && e.keyCode == KeyCode.A) return TimelineKeyAction.DeselectAll;
            if (e.control && e.keyCode == KeyCode.I) return TimelineKeyAction.InvertSelection;
            if (e.shift && e.keyCode == KeyCode.D) return TimelineKeyAction.Duplicate;
            if (e.control || e.alt || e.command) return TimelineKeyAction.None;

            return e.keyCode switch
            {
                KeyCode.A when !e.shift => TimelineKeyAction.SelectAll,
                KeyCode.B => TimelineKeyAction.BeginBoxSelect,
                KeyCode.G => TimelineKeyAction.BeginGrab,
                KeyCode.S => TimelineKeyAction.BeginScale,
                KeyCode.I => TimelineKeyAction.InsertKey,
                KeyCode.X or KeyCode.Delete or KeyCode.Backspace => TimelineKeyAction.Delete,
                KeyCode.Home => TimelineKeyAction.FrameAll,
                KeyCode.Period or KeyCode.KeypadPeriod => TimelineKeyAction.FrameSelected,
                KeyCode.LeftArrow => TimelineKeyAction.StepBack,
                KeyCode.RightArrow => TimelineKeyAction.StepForward,
                KeyCode.UpArrow => TimelineKeyAction.PreviousKey,
                KeyCode.DownArrow => TimelineKeyAction.NextKey,
                _ => TimelineKeyAction.None
            };
        }
    }
}
