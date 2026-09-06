using AnimationTools.Editor;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
    /// <summary>
    /// Which channel row a point in the tag lane falls in.
    /// </summary>
    /// <remarks>
    /// The distinction that matters is between "no row" and "ignore this click". Below the last row,
    /// in the lane's padding, and on a component with no channels at all are all no-row - and all
    /// three are exactly where a user reaches to start a box select. Treating -1 as a reason to
    /// return early is what made box select unreachable, and let the click fall through to the
    /// timeline and scrub the playhead instead.
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
