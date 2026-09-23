using AnimationTools.Editor;
using NUnit.Framework;

namespace AnimationTools.Tests
{
    /// <summary>
    /// Per-clip lane view state, and specifically what counts as a height the user chose.
    /// </summary>
    /// <remarks>
    /// Verifies that only a dragged height is restored as user-set: a user-set row stops taking
    /// <see cref="AnimationClipComponentTrack.RequestedLaneHeight"/>, so a merely stored height must
    /// not count.
    /// </remarks>
    public class ClipTrackViewStateTests
    {
        [Test]
        public void AHeightTheUserNeverDraggedIsNotRestored()
        {
            var state = new ClipTrackViewState();
            state.Record(new ClipTrackRow { LaneHeight = 68f, HeightIsUserSet = false });

            var row = new ClipTrackRow { LaneHeight = 24f };
            state.Apply(row);

            Assert.IsFalse(row.HeightIsUserSet);
            Assert.AreEqual(24f, row.LaneHeight);
        }

        [Test]
        public void AHeightTheUserDraggedSurvivesAndKeepsItsFlag()
        {
            var state = new ClipTrackViewState();
            state.Record(new ClipTrackRow { LaneHeight = 220f, HeightIsUserSet = true });

            var row = new ClipTrackRow { LaneHeight = 24f };
            state.Apply(row);

            Assert.IsTrue(row.HeightIsUserSet);
            Assert.AreEqual(220f, row.LaneHeight);
        }

        [Test]
        public void VisibilityRoundTrips()
        {
            var state = new ClipTrackViewState();
            state.Record(new ClipTrackRow { Visible = false });

            var row = new ClipTrackRow { Visible = true };
            state.Apply(row);

            Assert.IsFalse(row.Visible);
        }

        [Test]
        public void AnUnknownRowIsLeftAlone()
        {
            var row = new ClipTrackRow { LaneHeight = 44f, Visible = true };
            new ClipTrackViewState().Apply(row);

            Assert.AreEqual(44f, row.LaneHeight);
            Assert.IsTrue(row.Visible);
            Assert.IsFalse(row.HeightIsUserSet);
        }
    }
}
