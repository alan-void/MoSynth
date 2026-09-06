using System.Collections.Generic;
using AnimationTools.Editor;
using NUnit.Framework;

namespace AnimationTools.Tests
{
    /// <summary>
    /// The arithmetic behind every keyframe edit in the tag lane.
    /// </summary>
    /// <remarks>
    /// These exist because normalising can remove keys the caller asked for, and the track has to be
    /// able to tell what actually happened. Collisions are the interesting case: a moved key landing
    /// on a stationary one is not an error to reject but a cancellation to report, and a caller that
    /// assumed otherwise would keep a selection pointing at frames that no longer hold a key.
    /// </remarks>
    public class AnimationTagEditTests
    {
        private static List<int> Keys(params int[] frames) => new(frames);

        [Test]
        public void InsertAddsAKeyInOrder()
        {
            Assert.AreEqual(new[] { 2, 5, 9 },
                AnimationTagEdits.Insert(Keys(2, 9), 5, 100));
        }

        [Test]
        public void InsertingOnAnExistingKeyRemovesIt()
        {
            // The parity rule read as an edit: inserting where a key already sits is how you take
            // one out by hand, and it is the same operation a drag-onto performs.
            Assert.AreEqual(new[] { 2 }, AnimationTagEdits.Insert(Keys(2, 9), 9, 100));
        }

        [Test]
        public void DeleteRemovesOnlyTheNamedFrames()
        {
            Assert.AreEqual(new[] { 2, 14 },
                AnimationTagEdits.Delete(Keys(2, 9, 14), Keys(9), 100));
        }

        [Test]
        public void MoveShiftsTheSelectionAndLeavesTheRest()
        {
            Assert.AreEqual(new[] { 2, 12 },
                AnimationTagEdits.Move(Keys(2, 9), Keys(9), 3, 100));
        }

        [Test]
        public void AMovedKeyLandingOnAStationaryOneCancelsBoth()
        {
            // The signal is unchanged rather than holding a duplicate frame, which is what makes
            // frame-keyed selection safe.
            Assert.AreEqual(new[] { 2 }, AnimationTagEdits.Move(Keys(2, 9, 14), Keys(14), -5, 100));
        }

        [Test]
        public void MoveKeepsTheOrderWhenTheSelectionOvertakesTheRest()
        {
            Assert.AreEqual(new[] { 9, 20 }, AnimationTagEdits.Move(Keys(2, 9), Keys(2), 18, 100));
        }

        [Test]
        public void DuplicateAddsCopiesAtAnOffset()
        {
            Assert.AreEqual(new[] { 2, 3, 9, 10 },
                AnimationTagEdits.Duplicate(Keys(2, 9), Keys(2, 9), 1, 100));
        }

        [Test]
        public void ScaleSpreadsTheSelectionAboutThePivot()
        {
            Assert.AreEqual(new[] { 0, 20 },
                AnimationTagEdits.Scale(Keys(5, 15), Keys(5, 15), 10, 2f, 100));
        }

        [Test]
        public void ScaleLeavesUnselectedKeysAlone()
        {
            Assert.AreEqual(new[] { 0, 9 },
                AnimationTagEdits.Scale(Keys(5, 9), Keys(5), 10, 2f, 100));
        }

        [Test]
        public void ClampDeltaStopsTheGroupAtTheClipEdge()
        {
            // A group drag that ran off the end would silently lose the keys that left, so the
            // whole group stops instead.
            Assert.AreEqual(-4, AnimationTagEdits.ClampDelta(Keys(4, 9), -20, 100));
            Assert.AreEqual(6, AnimationTagEdits.ClampDelta(Keys(4, 94), 40, 100));
            Assert.AreEqual(3, AnimationTagEdits.ClampDelta(Keys(4, 9), 3, 100));
        }

        [Test]
        public void ClampDeltaOfAnEmptySelectionIsZero()
        {
            Assert.AreEqual(0, AnimationTagEdits.ClampDelta(Keys(), 12, 100));
        }
    }
}
