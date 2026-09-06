using System.Collections.Generic;
using GameplayTags;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
    /// <summary>
    /// A tag channel is a boolean signal stored as the frames it flips at, so every property that
    /// matters is a parity argument rather than an interval one.
    /// </summary>
    /// <remarks>
    /// Two of these carry real weight. Keys sharing a frame must cancel, because that is what a drag
    /// landing one key on another produces and the alternative is a duplicate frame the selection
    /// can no longer address. And a query must run down the hierarchy only: a clip annotated
    /// <c>action</c> has not claimed to be <c>action.walk</c>, and matching it would make every
    /// filter quietly too permissive.
    /// </remarks>
    public class AnimationTaggingTests
    {
        private readonly List<GameplayTagSO> _created = new();

        private GameplayTagSO Tag(string name, GameplayTagSO parent = null)
        {
            var tag = ScriptableObject.CreateInstance<GameplayTagSO>();
            tag.name = name;
            tag.UpdateName(name);
            if (parent) tag.SetParent(parent);

            _created.Add(tag);
            return tag;
        }

        [TearDown]
        public void TearDown()
        {
            foreach (var tag in _created) Object.DestroyImmediate(tag);
            _created.Clear();
        }

        private static AnimationTagging.TagChannel Channel(GameplayTagSO tag, params int[] toggles) =>
            new() { tag = tag, toggles = new List<int>(toggles) };

        private static List<int> Normalised(params int[] toggles)
        {
            var list = new List<int>(toggles);
            AnimationTagging.Normalise(list);
            return list;
        }

        // ---- normalising ----

        [Test]
        public void NormaliseSortsTheKeys()
        {
            Assert.AreEqual(new[] { 2, 5, 9 }, Normalised(9, 2, 5));
        }

        [Test]
        public void TwoKeysOnOneFrameCancel()
        {
            // A flip and an immediate flip back is the same signal as neither, and it is exactly
            // what dragging one key onto another produces.
            Assert.AreEqual(new[] { 4 }, Normalised(4, 7, 7));
        }

        [Test]
        public void AnOddRunOnOneFrameLeavesOneKey()
        {
            Assert.AreEqual(new[] { 7 }, Normalised(7, 7, 7));
        }

        [Test]
        public void KeysBeyondTheClipAreKeptAndOnlyNegativesDropped()
        {
            // The whole point of the frame space: a key past the end survives, because trimming a
            // clip must not destroy annotation. A negative one is corruption, not range - it would
            // make IsOn count for every frame from 0 up and invert the channel.
            Assert.AreEqual(new[] { 0, 10, 14 }, Normalised(-3, 0, 10, 14));
        }

        [Test]
        public void NormalisingAnEmptyListIsHarmless()
        {
            Assert.AreEqual(new int[0], Normalised());
        }

        // ---- reading the signal ----

        [Test]
        public void AKeySwitchesTheChannelAtItsOwnFrame()
        {
            var toggles = new List<int> { 5, 9 };

            Assert.IsFalse(AnimationTagging.IsOn(toggles, 4));
            Assert.IsTrue(AnimationTagging.IsOn(toggles, 5));
            Assert.IsTrue(AnimationTagging.IsOn(toggles, 8));
            Assert.IsFalse(AnimationTagging.IsOn(toggles, 9));
        }

        [Test]
        public void SpansPairConsecutiveKeys()
        {
            var spans = new List<AnimationTagging.TagSpan>();
            AnimationTagging.Spans(new List<int> { 2, 5, 10, 12 }, 20, spans);

            Assert.AreEqual(2, spans.Count);
            Assert.AreEqual(2, spans[0].StartFrame);
            Assert.AreEqual(5, spans[0].EndFrame);
            Assert.AreEqual(10, spans[1].StartFrame);
            Assert.AreEqual(12, spans[1].EndFrame);
        }

        [Test]
        public void ATrailingKeyRunsToTheEndOfTheClip()
        {
            // "On until the end" needs no closing key, so trimming a clip cannot turn an open span
            // into a closed one behind your back.
            var spans = new List<AnimationTagging.TagSpan>();
            AnimationTagging.Spans(new List<int> { 6 }, 20, spans);

            Assert.AreEqual(1, spans.Count);
            Assert.AreEqual(6, spans[0].StartFrame);
            Assert.AreEqual(20, spans[0].EndFrame);
        }

        // ---- querying ----

        private static List<AnimationClipSegment> Segments(GameplayTagQuery query, int frameCount,
            params AnimationTagging.TagChannel[] channels)
        {
            var results = new List<AnimationClipSegment>();
            AnimationTagging.FindSegments(null, channels, query, 0, frameCount, results);
            return results;
        }

        [Test]
        public void AQueryOnAParentMatchesAChannelTaggedBelowIt()
        {
            var action = Tag("action");
            var walk = Tag("walk", action);

            var segments = Segments(new GameplayTagQuery(all: new GameplayTagSet(new[] { action })),
                20, Channel(walk, 4, 12));

            Assert.AreEqual(1, segments.Count);
            Assert.AreEqual(4, segments[0].StartFrame);
            Assert.AreEqual(12, segments[0].EndFrame);
        }

        [Test]
        public void AQueryOnAChildDoesNotMatchAChannelTaggedOnlyWithTheParent()
        {
            var action = Tag("action");
            var walk = Tag("walk", action);

            Assert.AreEqual(0,
                Segments(new GameplayTagQuery(all: new GameplayTagSet(new[] { walk })), 20,
                    Channel(action, 4, 12)).Count);
        }

        [Test]
        public void AnExcludedTagCarvesTheSegmentUp()
        {
            var action = Tag("action");
            var walk = Tag("walk", action);
            var style = Tag("style");
            var tired = Tag("tired", style);

            var query = new GameplayTagQuery(
                all: new GameplayTagSet(new[] { action }),
                none: new GameplayTagSet(new[] { style }));

            var segments = Segments(query, 20, Channel(walk, 0, 20), Channel(tired, 5, 10));

            Assert.AreEqual(2, segments.Count);
            Assert.AreEqual(0, segments[0].StartFrame);
            Assert.AreEqual(5, segments[0].EndFrame);
            Assert.AreEqual(10, segments[1].StartFrame);
            Assert.AreEqual(20, segments[1].EndFrame);
        }

        [Test]
        public void TouchingMatchesMergeIntoOneSegment()
        {
            // Two channels that hand over at the same frame describe one continuous run, and
            // reporting it as two would make every consumer re-merge them.
            var action = Tag("action");
            var walk = Tag("walk", action);
            var run = Tag("run", action);

            var segments = Segments(new GameplayTagQuery(all: new GameplayTagSet(new[] { action })),
                20, Channel(walk, 0, 8), Channel(run, 8, 16));

            Assert.AreEqual(1, segments.Count);
            Assert.AreEqual(0, segments[0].StartFrame);
            Assert.AreEqual(16, segments[0].EndFrame);
        }

        [Test]
        public void AnEmptyQueryReturnsTheWholeSliceAsOneSegment()
        {
            var segments = Segments(new GameplayTagQuery(), 20, Channel(Tag("action"), 4, 12));

            Assert.AreEqual(1, segments.Count);
            Assert.AreEqual(0, segments[0].StartFrame);
            Assert.AreEqual(20, segments[0].EndFrame);
        }

        [Test]
        public void AnyMatchesWhenEitherChannelIsOn()
        {
            var walk = Tag("walk");
            var run = Tag("run");

            var query = new GameplayTagQuery(any: new GameplayTagSet(new[] { walk, run }));
            var segments = Segments(query, 20, Channel(walk, 2, 6), Channel(run, 12, 15));

            Assert.AreEqual(2, segments.Count);
            Assert.AreEqual(2, segments[0].StartFrame);
            Assert.AreEqual(6, segments[0].EndFrame);
            Assert.AreEqual(12, segments[1].StartFrame);
            Assert.AreEqual(15, segments[1].EndFrame);
        }

        [Test]
        public void AChannelWithNoTagIsIgnoredRatherThanCrashing()
        {
            var action = Tag("action");

            var segments = Segments(new GameplayTagQuery(all: new GameplayTagSet(new[] { action })),
                20, Channel(null, 0, 20), Channel(action, 3, 7));

            Assert.AreEqual(1, segments.Count);
            Assert.AreEqual(3, segments[0].StartFrame);
            Assert.AreEqual(7, segments[0].EndFrame);
        }

        [Test]
        public void AnOpenChannelMatchesToTheEndOfTheClip()
        {
            var action = Tag("action");

            var segments = Segments(new GameplayTagQuery(all: new GameplayTagSet(new[] { action })),
                20, Channel(action, 6));

            Assert.AreEqual(1, segments.Count);
            Assert.AreEqual(6, segments[0].StartFrame);
            Assert.AreEqual(20, segments[0].EndFrame);
        }
    }
}
