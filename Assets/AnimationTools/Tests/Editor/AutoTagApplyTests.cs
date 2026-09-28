using System.Collections.Generic;
using AnimationTools.Editor;
using NUnit.Framework;

namespace AnimationTools.Tests
{
public class AutoTagApplyTests
{
    [Test]
    public void NoSegmentsMakeNoToggles()
    {
        CollectionAssert.IsEmpty(AutoTagApply.SegmentsToToggles(new List<(int, int)>()));
        CollectionAssert.IsEmpty(AutoTagApply.SegmentsToToggles(null));
    }

    [Test]
    public void SegmentsBecomeStartEndPairs()
    {
        var toggles = AutoTagApply.SegmentsToToggles(new[] { (12, 140), (200, 260) });

        CollectionAssert.AreEqual(new[] { 12, 140, 200, 260 }, toggles);
    }

    [Test]
    public void UnsortedSegmentsAreSorted()
    {
        var toggles = AutoTagApply.SegmentsToToggles(new[] { (200, 260), (0, 10), (40, 50) });

        CollectionAssert.AreEqual(new[] { 0, 10, 40, 50, 200, 260 }, toggles);
    }

    [Test]
    public void AdjacentSegmentsMerge()
    {
        // Keys at 10 and 10 would cancel under Normalise, so touching segments are joined explicitly.
        var toggles = AutoTagApply.SegmentsToToggles(new[] { (0, 10), (10, 20) });

        CollectionAssert.AreEqual(new[] { 0, 20 }, toggles);
    }

    [Test]
    public void OverlappingAndContainedSegmentsMerge()
    {
        var toggles = AutoTagApply.SegmentsToToggles(new[] { (0, 30), (5, 10), (25, 40), (50, 60) });

        CollectionAssert.AreEqual(new[] { 0, 40, 50, 60 }, toggles);
    }

    [Test]
    public void EmptyAndInvertedSegmentsAreDropped()
    {
        var toggles = AutoTagApply.SegmentsToToggles(new[] { (5, 5), (9, 3), (20, 30) });

        CollectionAssert.AreEqual(new[] { 20, 30 }, toggles);
    }

    [Test]
    public void TogglesSurviveNormalise()
    {
        var toggles = AutoTagApply.SegmentsToToggles(new[] { (30, 40), (0, 10), (10, 15) });
        var normalised = new List<int>(toggles);
        AnimationTagging.Normalise(normalised);

        CollectionAssert.AreEqual(toggles, normalised);
    }

    [Test]
    public void ManifestRoundTripsWithCamelCaseKeys()
    {
        var manifest = new AutoTagManifest
        {
            Model = "gemini-3.8-flash",
            MediaResolution = "low",
            MinConfidence = 0.5f,
            MinSpanFrames = 3,
            MinGapFrames = 4,
            PerTagRequests = true,
            Tags =
            {
                new AutoTagManifest.TagEntry
                {
                    Name = "animation.action.walking",
                    Description = "Character walks",
                    Ancestors = { new AutoTagManifest.TagAncestor { Name = "animation.action", Description = "Body" } }
                }
            },
            Clips =
            {
                new AutoTagManifest.ClipEntry
                {
                    Guid = "0123abcd",
                    Name = "walk_01",
                    AssetPath = "Assets/Clips/walk_01.asset",
                    ClipFrameRate = 30f,
                    StartFrame = 0,
                    EndFrame = 450,
                    RenderFps = 10f,
                    FrameStep = 3,
                    Video = "videos/0123abcd.mp4",
                    GroundSpeed = { 1.2f, 1.4f }
                }
            }
        };

        var json = manifest.ToJson();

        foreach (var key in new[]
                 {
                     "model", "mediaResolution", "minConfidence", "minSpanFrames", "minGapFrames", "perTagRequests", "tags",
                     "name", "description", "ancestors", "clips", "guid", "assetPath", "clipFrameRate",
                     "startFrame", "endFrame", "renderFps", "frameStep", "video", "groundSpeed"
                 })
        {
            StringAssert.Contains($"\"{key}\":", json, key);
        }

        var read = AutoTagManifest.FromJson(json);
        Assert.AreEqual(manifest.Model, read.Model);
        Assert.AreEqual(4, read.MinGapFrames);
        Assert.IsTrue(read.PerTagRequests);
        Assert.AreEqual("animation.action", read.Tags[0].Ancestors[0].Name);

        var clip = read.Clips[0];
        Assert.AreEqual("0123abcd", clip.Guid);
        Assert.AreEqual(450, clip.EndFrame);
        Assert.AreEqual(3, clip.FrameStep);
        Assert.AreEqual("videos/0123abcd.mp4", clip.Video);
        CollectionAssert.AreEqual(new[] { 1.2f, 1.4f }, clip.GroundSpeed);
    }

    [Test]
    public void ResultsReadTheContractShape()
    {
        const string json = @"{
          ""model"": ""gemini-3.8-flash"",
          ""clips"": [
            {
              ""guid"": ""aaaa"",
              ""error"": null,
              ""tags"": [
                {
                  ""name"": ""animation.action.walking"",
                  ""reasoning"": ""short text"",
                  ""segments"": [ { ""startFrame"": 12, ""endFrame"": 140, ""confidence"": 0.9 } ]
                }
              ]
            },
            { ""guid"": ""bbbb"", ""error"": ""timed out"", ""tags"": [] }
          ]
        }";

        var results = AutoTagResults.FromJson(json);

        Assert.AreEqual(2, results.Clips.Count);
        Assert.IsNull(results.Clips[0].Error);
        Assert.AreEqual("timed out", results.Clips[1].Error);

        var segment = results.Clips[0].Tags[0].Segments[0];
        Assert.AreEqual(12, segment.StartFrame);
        Assert.AreEqual(140, segment.EndFrame);
        Assert.AreEqual(0.9f, segment.Confidence, 1e-6f);

        var written = results.ToJson();
        StringAssert.Contains("\"error\": null", written);
        StringAssert.Contains("\"startFrame\": 12", written);
        Assert.AreEqual("short text", AutoTagResults.FromJson(written).Clips[0].Tags[0].Reasoning);
    }

    [Test]
    public void GroundSpeedIsAveragedPerSecond()
    {
        var perFrame = new[] { 1f, 1f, 3f, 3f, 5f };

        var perSecond = AutoTagManifest.GroundSpeedPerSecond(perFrame, 2f);

        CollectionAssert.AreEqual(new[] { 1f, 3f, 5f }, perSecond);
    }

    [TestCase(30f, 10f, 3)]
    [TestCase(24f, 10f, 2)]
    [TestCase(60f, 10f, 6)]
    [TestCase(5f, 10f, 1)]
    public void FrameStepIsTheNearestWholeStep(float clipFrameRate, float renderFps, int expected)
    {
        Assert.AreEqual(expected, ClipVideoRenderer.FrameStep(clipFrameRate, renderFps));
    }

    [Test]
    public void VideoFrameCountCoversTheSlice()
    {
        Assert.AreEqual(150, ClipVideoRenderer.VideoFrameCount(450, 3));
        Assert.AreEqual(151, ClipVideoRenderer.VideoFrameCount(451, 3));
        Assert.AreEqual(0, ClipVideoRenderer.VideoFrameCount(0, 3));
    }

    [Test]
    public void ProtocolLinesParse()
    {
        var progress = AutoTagAnnotateProcess.ProtocolLine.Parse("PROGRESS 2 10 walk 01 fast");
        Assert.AreEqual(AutoTagAnnotateProcess.ProtocolLine.LineKind.Progress, progress.Kind);
        Assert.AreEqual(2, progress.Index);
        Assert.AreEqual(10, progress.Total);
        Assert.AreEqual("walk 01 fast", progress.Text);

        var usage = AutoTagAnnotateProcess.ProtocolLine.Parse("USAGE abcd 12000 340");
        Assert.AreEqual(AutoTagAnnotateProcess.ProtocolLine.LineKind.Usage, usage.Kind);
        Assert.AreEqual("abcd", usage.Guid);
        Assert.AreEqual(12000, usage.InputTokens);
        Assert.AreEqual(340, usage.OutputTokens);

        var error = AutoTagAnnotateProcess.ProtocolLine.Parse("CLIPERROR abcd model said no: twice");
        Assert.AreEqual(AutoTagAnnotateProcess.ProtocolLine.LineKind.ClipError, error.Kind);
        Assert.AreEqual("model said no: twice", error.Text);

        var done = AutoTagAnnotateProcess.ProtocolLine.Parse("DONE 9 1");
        Assert.AreEqual(AutoTagAnnotateProcess.ProtocolLine.LineKind.Done, done.Kind);
        Assert.AreEqual(9, done.Index);
        Assert.AreEqual(1, done.Total);

        foreach (var other in new[] { "", "hello", "PROGRESS x y", "USAGE abcd" })
        {
            Assert.AreEqual(AutoTagAnnotateProcess.ProtocolLine.LineKind.Other,
                AutoTagAnnotateProcess.ProtocolLine.Parse(other).Kind, other);
        }
    }

    [Test]
    public void SettingsParseClampsAndDefaults()
    {
        var defaults = AutoTagSettings.Parse("not json");
        Assert.AreEqual("gemini-3.8-flash", defaults.model);
        Assert.AreEqual(0.5f, defaults.minConfidence);

        var clamped = AutoTagSettings.Parse("{\"mediaResolution\": \"ultra\", \"minConfidence\": 3, \"renderFps\": 0}");
        Assert.AreEqual(AutoTagSettings.LowResolution, clamped.mediaResolution);
        Assert.AreEqual(1f, clamped.minConfidence);
        Assert.AreEqual(1f, clamped.renderFps);
    }
}
}
