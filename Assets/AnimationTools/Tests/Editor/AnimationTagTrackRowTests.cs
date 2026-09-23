using AnimationTools.Editor;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
    /// <summary>
    /// Which channel row a point in the tag lane falls in.
    /// </summary>
    /// <remarks>
    /// Verifies that padding, the space below the last row, and a channel-less component all report
    /// no row (-1) — the empty space where a box select starts.
    /// </remarks>
    public class AnimationTagTrackRowTests
    {
        // Matches the track's own row metrics: 2px padding, then 20px rows.
        private static readonly Rect Lane = new(0f, 100f, 400f, 64f);

        [Test]
        public void APointInsideARowPicksThatRow()
        {
            Assert.AreEqual(0, AnimationTagTrack.RowAt(Lane, 105f, 3));
            Assert.AreEqual(1, AnimationTagTrack.RowAt(Lane, 125f, 3));
            Assert.AreEqual(2, AnimationTagTrack.RowAt(Lane, 145f, 3));
        }

        [Test]
        public void BelowTheLastRowIsEmptySpace()
        {
            Assert.AreEqual(-1, AnimationTagTrack.RowAt(Lane, 145f, 2));
        }

        [Test]
        public void TheTopPaddingIsEmptySpace()
        {
            Assert.AreEqual(-1, AnimationTagTrack.RowAt(Lane, 100.5f, 3));
        }

        [Test]
        public void AComponentWithNoChannelsIsAllEmptySpace()
        {
            Assert.AreEqual(-1, AnimationTagTrack.RowAt(Lane, 105f, 0));
            Assert.AreEqual(-1, AnimationTagTrack.RowAt(Lane, 150f, 0));
        }
    }
}
