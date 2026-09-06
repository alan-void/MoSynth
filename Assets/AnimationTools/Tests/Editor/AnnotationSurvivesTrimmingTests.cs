using System.Collections.Generic;
using GameplayTags;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
    /// <summary>
    /// Trimming a clip must not touch its annotation.
    /// </summary>
    /// <remarks>
    /// This is the regression these two components were rewritten for. `OnValidate` used to delete
    /// every anchor and key outside the current `[startFrame, endFrame)`, so dragging the slice
    /// handle in and back out destroyed the annotation between — permanently, silently, and on a
    /// clip whose range ended up exactly where it started. It cost a real clip 44 hand-corrected
    /// footfalls.
    /// <para>
    /// The frames are clip-local now, so a trim changes which frames are extracted and nothing else.
    /// These tests drive `OnValidate` directly rather than through the editor, because that is the
    /// method that did the deleting.
    /// </para>
    /// </remarks>
    public class AnnotationSurvivesTrimmingTests
    {
        private readonly List<Object> _created = new();

        [TearDown]
        public void TearDown()
        {
            foreach (var asset in _created) Object.DestroyImmediate(asset);
            _created.Clear();
        }

        private AnnotatedAnimationClip Clip(int startFrame, int endFrame)
        {
            var clip = ScriptableObject.CreateInstance<AnnotatedAnimationClip>();
            clip.startFrame = startFrame;
            clip.endFrame = endFrame;

            _created.Add(clip);
            return clip;
        }

        private GameplayTagSO Tag(string name)
        {
            var tag = ScriptableObject.CreateInstance<GameplayTagSO>();
            tag.name = name;
            tag.UpdateName(name);

            _created.Add(tag);
            return tag;
        }

        private static GaitPhaseComponent Gait(params int[] frames)
        {
            var component = new GaitPhaseComponent();
            foreach (var frame in frames)
            {
                component.footfalls.Add(new GaitPhase.Footfall(frame, GaitPhase.Foot.Left));
            }

            return component;
        }

        private static List<int> Frames(GaitPhaseComponent component)
        {
            var frames = new List<int>();
            foreach (var footfall in component.footfalls) frames.Add(footfall.frame);

            return frames;
        }

        [Test]
        public void PullingTheEndInKeepsTheAnchorsBeyondIt()
        {
            var clip = Clip(0, 100);
            var gait = Gait(10, 40, 70, 95);

            gait.OnValidate(clip);
            Assert.AreEqual(new[] { 10, 40, 70, 95 }, Frames(gait));

            clip.endFrame = 50;
            gait.OnValidate(clip);

            Assert.AreEqual(new[] { 10, 40, 70, 95 }, Frames(gait),
                "anchors past the new end were deleted; a trim must not destroy annotation");
        }

        [Test]
        public void PushingTheEndBackOutRestoresNothingBecauseNothingWasLost()
        {
            var clip = Clip(0, 100);
            var gait = Gait(10, 40, 70, 95);

            clip.endFrame = 50;
            gait.OnValidate(clip);

            clip.endFrame = 100;
            gait.OnValidate(clip);

            Assert.AreEqual(new[] { 10, 40, 70, 95 }, Frames(gait));
        }

        [Test]
        public void MovingTheStartDoesNotMoveTheAnchors()
        {
            // Clip-local is the whole reason this holds. Under slice-local addressing the stored
            // numbers would not change either, but each would name a moment 100 frames later in the
            // animation - a silent edit with nothing in the asset to show for it.
            var clip = Clip(0, 200);
            var gait = Gait(10, 40, 70);

            clip.startFrame = 100;
            gait.OnValidate(clip);

            Assert.AreEqual(new[] { 10, 40, 70 }, Frames(gait));
        }

        [Test]
        public void TrimmingKeepsTagKeysBeyondTheEnd()
        {
            var clip = Clip(0, 100);
            var tags = new AnimationTagComponent();
            tags.channels.Add(new AnimationTagging.TagChannel
            {
                tag = Tag("action"), toggles = new List<int> { 10, 40, 70, 95 }
            });

            clip.endFrame = 50;
            tags.OnValidate(clip);

            Assert.AreEqual(new[] { 10, 40, 70, 95 }, tags.channels[0].toggles);
        }

        [Test]
        public void ValidatingStillFoldsChannelsThatNameOneTag()
        {
            // The integrity rules stay; only the range clamp went.
            var clip = Clip(0, 100);
            var tag = Tag("action");
            var tags = new AnimationTagComponent();
            tags.channels.Add(new AnimationTagging.TagChannel { tag = tag, toggles = new List<int> { 10 } });
            tags.channels.Add(new AnimationTagging.TagChannel { tag = tag, toggles = new List<int> { 40 } });

            tags.OnValidate(clip);

            Assert.AreEqual(1, tags.channels.Count);
            Assert.AreEqual(new[] { 10, 40 }, tags.channels[0].toggles);
        }

        [Test]
        public void PhaseIgnoresAnchorsOutsideTheClipWithoutThrowing()
        {
            // What made the deletion unnecessary in the first place: Evaluate filters these itself.
            var phase = new float[50];
            var rate = new float[50];

            Assert.DoesNotThrow(() => GaitPhase.Evaluate(
                new List<GaitPhase.Footfall>
                {
                    new(-20, GaitPhase.Foot.Left),
                    new(10, GaitPhase.Foot.Right),
                    new(30, GaitPhase.Foot.Left),
                    new(900, GaitPhase.Foot.Right)
                },
                1f / 30f, phase, rate));

            Assert.Greater(rate[20], 0f, "the two in-range anchors should still define a cycle");
        }
    }
}
