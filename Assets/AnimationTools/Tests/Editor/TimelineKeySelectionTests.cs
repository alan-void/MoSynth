using System.Collections.Generic;
using AnimationTools.Editor;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
    /// <summary>
    /// Selecting keyframes by the frame they sit on rather than their index in a list.
    /// </summary>
    /// <remarks>
    /// The reason for the whole design: every edit re-sorts its list, so an index means something
    /// different afterwards and an index-keyed selection has to be thrown away after every move. A
    /// frame is unique within a row, so it survives - and a key that cancelled against another
    /// leaves the selection by simply no longer existing.
    /// </remarks>
    public class TimelineKeySelectionTests
    {
        [Test]
        public void ASelectionSurvivesAMoveThatResortsTheList()
        {
            var selection = new TimelineKeySelection();
            selection.Add(0, 2);

            var moved = AnimationTagEdits.Move(new List<int> { 2, 9 }, new List<int> { 2 }, 18, 100);
            var frames = new List<int> { 2 + 18 };

            selection.ReplaceRow(0, frames);
            selection.Intersect(0, moved);

            Assert.AreEqual(new[] { 9, 20 }, moved);
            Assert.AreEqual(1, selection.Count);
            Assert.IsTrue(selection.Contains(0, 20));
        }

        [Test]
        public void AKeyThatCancelledLeavesTheSelection()
        {
            var selection = new TimelineKeySelection();
            selection.Add(0, 14);

            var moved = AnimationTagEdits.Move(new List<int> { 2, 9, 14 }, new List<int> { 14 }, -5,
                100);

            selection.ReplaceRow(0, new List<int> { 9 });
            selection.Intersect(0, moved);

            Assert.AreEqual(new[] { 2 }, moved);
            Assert.AreEqual(0, selection.Count);
        }

        [Test]
        public void ReplacingOneRowLeavesTheOthersAlone()
        {
            var selection = new TimelineKeySelection();
            selection.Add(0, 4);
            selection.Add(1, 7);

            selection.ReplaceRow(0, new List<int> { 12 });

            Assert.IsTrue(selection.Contains(1, 7));
            Assert.IsTrue(selection.Contains(0, 12));
            Assert.IsFalse(selection.Contains(0, 4));
        }

        [Test]
        public void ToggleAddsThenRemovesTheSameKey()
        {
            var selection = new TimelineKeySelection();

            selection.Toggle(0, 5);
            Assert.IsTrue(selection.Contains(0, 5));

            selection.Toggle(0, 5);
            Assert.IsFalse(selection.Contains(0, 5));
        }

        [Test]
        public void FramesInReturnsOneRowSorted()
        {
            var selection = new TimelineKeySelection();
            selection.Add(0, 9);
            selection.Add(0, 2);
            selection.Add(1, 5);

            var frames = new List<int>();
            selection.FramesIn(0, frames);

            Assert.AreEqual(new[] { 2, 9 }, frames);
        }

        [Test]
        public void TheSameFrameInTwoRowsIsTwoDifferentKeys()
        {
            var selection = new TimelineKeySelection();
            selection.Add(0, 5);

            Assert.IsFalse(selection.Contains(1, 5));
            Assert.AreEqual(1, selection.Count);
        }

        // ---- hit testing ----

        [Test]
        public void PickTakesTheNearestKeyWithinTolerance()
        {
            var frames = new List<int> { 0, 10, 20 };

            Assert.AreEqual(10, TimelineKeyHitTester.Pick(frames, 101f, f => f * 10f));
            Assert.AreEqual(-1, TimelineKeyHitTester.Pick(frames, 150f, f => f * 10f));
        }

        [Test]
        public void PickRangeTakesEveryKeyInsideTheBand()
        {
            var frames = new List<int> { 0, 10, 20, 30 };
            var picked = new List<int>();

            TimelineKeyHitTester.PickRange(frames, 95f, 205f, f => f * 10f, picked);

            Assert.AreEqual(new[] { 10, 20 }, picked);
        }

        [Test]
        public void PickRangeAcceptsABoxDraggedRightToLeft()
        {
            var frames = new List<int> { 0, 10, 20 };
            var picked = new List<int>();

            TimelineKeyHitTester.PickRange(frames, 205f, 95f, f => f * 10f, picked);

            Assert.AreEqual(new[] { 10, 20 }, picked);
        }

        // ---- the modal operator ----

        private static TimelineModalOperator Grab(float pixelsPerFrame = 10f)
        {
            var operatorUnderTest = new TimelineModalOperator();
            operatorUnderTest.Begin(TimelineModalKind.Grab, 0, new Vector2(100f, 10f), pixelsPerFrame,
                0, 100f);

            return operatorUnderTest;
        }

        [Test]
        public void MouseTravelBecomesAFrameDelta()
        {
            var grab = Grab();
            grab.HandleEvent(new Event { type = EventType.MouseMove, mousePosition = new Vector2(140f, 10f) });

            Assert.AreEqual(4, grab.FrameDelta);
        }

        [Test]
        public void TypingANumberOverridesTheMouse()
        {
            var grab = Grab();
            grab.HandleEvent(new Event { type = EventType.MouseMove, mousePosition = new Vector2(140f, 10f) });
            grab.HandleEvent(new Event { type = EventType.KeyDown, character = '1' });
            grab.HandleEvent(new Event { type = EventType.KeyDown, character = '2' });

            Assert.AreEqual(12, grab.FrameDelta);
            Assert.IsTrue(grab.StatusText.Contains("12"));
        }

        [Test]
        public void EscapeCancelsAndDiscardsTheDelta()
        {
            // The whole reason a mode exists: a cancelled edit writes nothing, so it costs neither
            // an undo entry nor the clip re-bake a commit triggers.
            var grab = Grab();
            grab.HandleEvent(new Event { type = EventType.MouseMove, mousePosition = new Vector2(200f, 10f) });

            var result = grab.HandleEvent(new Event { type = EventType.KeyDown, keyCode = KeyCode.Escape });

            Assert.AreEqual(TimelineModalResult.Cancelled, result);
            Assert.AreEqual(0, grab.FrameDelta);
            Assert.IsFalse(grab.IsActive);
        }

        [Test]
        public void AClickConfirmsAndKeepsTheDelta()
        {
            var grab = Grab();
            grab.HandleEvent(new Event { type = EventType.MouseMove, mousePosition = new Vector2(200f, 10f) });

            var result = grab.HandleEvent(new Event
            {
                type = EventType.MouseDown, button = 0, mousePosition = new Vector2(200f, 10f)
            });

            Assert.AreEqual(TimelineModalResult.Confirmed, result);
            Assert.AreEqual(10, grab.FrameDelta);
            Assert.IsFalse(grab.IsActive);
        }

        [Test]
        public void ARepaintFollowsTheCursorAndIsNotConsumed()
        {
            // The clip editor is a UI Toolkit window, so MouseMove with no button held may never
            // arrive; sampling repaints is what actually makes a grab track the mouse. Consuming a
            // repaint would blank the lane it was meant to draw.
            var grab = Grab();
            var repaint = new Event { type = EventType.Repaint, mousePosition = new Vector2(140f, 10f) };

            var result = grab.HandleEvent(repaint);

            Assert.AreEqual(TimelineModalResult.Updated, result);
            Assert.AreEqual(4, grab.FrameDelta);
            Assert.AreEqual(EventType.Repaint, repaint.type);
        }

        [Test]
        public void ARepaintThatDidNotMoveTheCursorReportsNothing()
        {
            // Repaints arrive continuously while a mode is up, so a still cursor must not look like
            // a change and drive an endless repaint loop of its own.
            var grab = Grab();
            grab.HandleEvent(new Event { type = EventType.Repaint, mousePosition = new Vector2(140f, 10f) });

            Assert.AreEqual(TimelineModalResult.None,
                grab.HandleEvent(new Event { type = EventType.Repaint, mousePosition = new Vector2(140f, 10f) }));
        }

        [Test]
        public void AConfirmedModeStillReportsWhatItWas()
        {
            // The owner reads Kind in the handler that applies the edit. Clearing it as part of
            // finishing made every confirmed grab, scale and box select apply nothing at all - the
            // keys drew at the previewed position and then snapped back.
            var grab = Grab();
            grab.HandleEvent(new Event { type = EventType.MouseMove, mousePosition = new Vector2(200f, 10f) });

            var result = grab.HandleEvent(new Event
            {
                type = EventType.MouseDown, button = 0, mousePosition = new Vector2(200f, 10f)
            });

            Assert.AreEqual(TimelineModalResult.Confirmed, result);
            Assert.AreEqual(TimelineModalKind.Grab, grab.Kind);
            Assert.AreEqual(10, grab.FrameDelta);
            Assert.IsFalse(grab.IsActive);
        }

        [Test]
        public void CancellingClearsTheKindAFinishedModeLeftBehind()
        {
            var grab = Grab();
            grab.HandleEvent(new Event { type = EventType.MouseDown, button = 0, mousePosition = new Vector2(200f, 10f) });

            grab.Cancel();

            Assert.AreEqual(TimelineModalKind.None, grab.Kind);
            Assert.IsFalse(grab.IsActive);
        }

        [Test]
        public void AFinishedModeIgnoresFurtherEvents()
        {
            var grab = Grab();
            grab.HandleEvent(new Event { type = EventType.MouseDown, button = 0, mousePosition = new Vector2(200f, 10f) });

            Assert.AreEqual(TimelineModalResult.None,
                grab.HandleEvent(new Event { type = EventType.MouseMove, mousePosition = new Vector2(400f, 10f) }));
            Assert.AreEqual(10, grab.FrameDelta);
        }

        [Test]
        public void ABoxIsTheRectBetweenTheTwoCorners()
        {
            var box = new TimelineModalOperator();
            box.Begin(TimelineModalKind.BoxSelect, 0, new Vector2(200f, 40f), 10f, 0, 200f);
            box.HandleEvent(new Event { type = EventType.MouseDrag, mousePosition = new Vector2(100f, 10f) });

            Assert.AreEqual(Rect.MinMaxRect(100f, 10f, 200f, 40f), box.BoxRect);
        }
    }
}
