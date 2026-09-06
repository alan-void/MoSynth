using AnimationTools.Editor;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
    /// <summary>
    /// The key-to-action table every keyframe lane shares.
    /// </summary>
    /// <remarks>
    /// Worth testing because a binding that resolves to the wrong action and one that resolves to
    /// nothing look identical from inside the editor - you press a key and nothing happens. The
    /// modifier cases are the ones that actually bite: read in the wrong order, <c>Alt+A</c> selects
    /// everything instead of clearing it, and <c>Shift+D</c> deletes instead of duplicating.
    /// </remarks>
    public class TimelineKeymapTests
    {
        private static TimelineKeyAction Resolve(KeyCode key, bool alt = false, bool control = false,
            bool shift = false) =>
            TimelineKeymap.Resolve(new Event
            {
                type = EventType.KeyDown, keyCode = key, alt = alt, control = control, shift = shift
            });

        [Test]
        public void ModifiedKeysAreReadBeforeTheirBareForm()
        {
            Assert.AreEqual(TimelineKeyAction.SelectAll, Resolve(KeyCode.A));
            Assert.AreEqual(TimelineKeyAction.DeselectAll, Resolve(KeyCode.A, alt: true));

            Assert.AreEqual(TimelineKeyAction.InsertKey, Resolve(KeyCode.I));
            Assert.AreEqual(TimelineKeyAction.InvertSelection, Resolve(KeyCode.I, control: true));

            Assert.AreEqual(TimelineKeyAction.Duplicate, Resolve(KeyCode.D, shift: true));
        }

        [Test]
        public void AnUnboundModifierCombinationResolvesToNothing()
        {
            // Ctrl+G and the like belong to whatever else wants them, not to a half-matched grab.
            Assert.AreEqual(TimelineKeyAction.None, Resolve(KeyCode.G, control: true));
            Assert.AreEqual(TimelineKeyAction.None, Resolve(KeyCode.B, alt: true));
        }

        [Test]
        public void EveryDeleteSpellingResolves()
        {
            Assert.AreEqual(TimelineKeyAction.Delete, Resolve(KeyCode.X));
            Assert.AreEqual(TimelineKeyAction.Delete, Resolve(KeyCode.Delete));
            Assert.AreEqual(TimelineKeyAction.Delete, Resolve(KeyCode.Backspace));
        }

        [Test]
        public void TheViewKeysResolveSoATrackCanDeclineThem()
        {
            // Home and "." are the timeline's, because a track only sees a copy of the axis. They
            // still resolve here so the inspector's summary can list them.
            Assert.AreEqual(TimelineKeyAction.FrameAll, Resolve(KeyCode.Home));
            Assert.AreEqual(TimelineKeyAction.FrameSelected, Resolve(KeyCode.Period));
        }

        [Test]
        public void OnlyKeyDownResolves()
        {
            Assert.AreEqual(TimelineKeyAction.None,
                TimelineKeymap.Resolve(new Event { type = EventType.KeyUp, keyCode = KeyCode.G }));
            Assert.AreEqual(TimelineKeyAction.None, TimelineKeymap.Resolve(null));
        }

        [Test]
        public void EveryBindingInTheSummaryIsReachable()
        {
            // The summary is what tells a user the keymap exists, so a line describing a binding
            // that resolves to nothing is worse than no line at all.
            Assert.AreEqual(TimelineKeyAction.BeginBoxSelect, Resolve(KeyCode.B));
            Assert.AreEqual(TimelineKeyAction.BeginGrab, Resolve(KeyCode.G));
            Assert.AreEqual(TimelineKeyAction.BeginScale, Resolve(KeyCode.S));
            Assert.AreEqual(TimelineKeyAction.PreviousKey, Resolve(KeyCode.UpArrow));
            Assert.AreEqual(TimelineKeyAction.NextKey, Resolve(KeyCode.DownArrow));
            Assert.AreEqual(TimelineKeyAction.StepBack, Resolve(KeyCode.LeftArrow));
            Assert.AreEqual(TimelineKeyAction.StepForward, Resolve(KeyCode.RightArrow));
        }
    }
}
