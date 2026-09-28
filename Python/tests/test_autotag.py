"""
The auto-tag pipeline short of the model call: segment post-processing, the prompt and schema, the
run-folder JSON, and the annotate driver with the model call injected or answered from ``raw/``.
None of it imports the Gemini SDK.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotag import annotate  # noqa: E402
from autotag.manifest import (  # noqa: E402
    ClipResult, Results, Segment, TagResult, load_manifest, write_results)
from autotag.postprocess import clean_segments, postprocess_clip  # noqa: E402
from autotag.prompt import build_prompt, response_schema  # noqa: E402

WALK = "animation.action.walking"
IDLE = "animation.action.idle"
GUID = "0123abcd"

MANIFEST = {
    "model": "gemini-3.8-flash",
    "mediaResolution": "low",
    "minConfidence": 0.5,
    "minSpanFrames": 3,
    "minGapFrames": 3,
    "tags": [
        {"name": WALK, "description": "Character walks at a steady pace",
         "ancestors": [{"name": "animation.action",
                        "description": "What the character's body is doing"}]},
        {"name": IDLE, "description": "Character stands still", "ancestors": []},
    ],
    "clips": [{
        "guid": GUID, "name": "walk_01", "assetPath": "Assets/Clips/walk_01.asset",
        "clipFrameRate": 30.0, "startFrame": 10, "endFrame": 100, "renderFps": 10.0,
        "frameStep": 3, "video": f"videos/{GUID}.mp4", "groundSpeed": [1.2, 1.4, 0.1],
    }],
}

MODEL_ANSWER = {"tags": [
    {"name": WALK, "reasoning": "It walks.",
     "segments": [{"startFrame": 0, "endFrame": 40, "confidence": 0.9},
                  {"startFrame": 42, "endFrame": 60, "confidence": 0.7}]},
    {"name": "animation.action.flying", "reasoning": "Invented.",
     "segments": [{"startFrame": 10, "endFrame": 20, "confidence": 1.0}]},
]}


def clean(segments, start=0, end=100, min_confidence=0.5, min_span=3, min_gap=3):
    return clean_segments([Segment(*s) for s in segments], start, end,
                          min_confidence, min_span, min_gap)


def write_run_folder(folder: str, **overrides) -> None:
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({**MANIFEST, **overrides}, f)


class CleanSegmentsTest(unittest.TestCase):

    def test_drops_low_confidence(self):
        self.assertEqual(clean([(10, 20, 0.4), (30, 40, 0.5)]), [Segment(30, 40, 0.5)])

    def test_clamps_to_slice_and_drops_empty(self):
        self.assertEqual(clean([(-5, 20, 0.9), (90, 120, 0.9), (150, 160, 0.9)], end=100),
                         [Segment(0, 20, 0.9), Segment(90, 100, 0.9)])

    def test_merges_small_gap_with_max_confidence(self):
        self.assertEqual(clean([(22, 30, 0.9), (10, 20, 0.6)]), [Segment(10, 30, 0.9)])

    def test_keeps_gap_at_threshold(self):
        self.assertEqual(clean([(10, 20, 0.9), (23, 30, 0.9)]),
                         [Segment(10, 20, 0.9), Segment(23, 30, 0.9)])

    def test_merges_adjacent_and_overlapping_even_without_gap_threshold(self):
        self.assertEqual(clean([(10, 20, 0.9), (20, 30, 0.8), (25, 35, 0.7)], min_gap=0),
                         [Segment(10, 35, 0.9)])

    def test_drops_short_spans_after_merging(self):
        self.assertEqual(clean([(10, 12, 0.9), (13, 14, 0.9), (50, 52, 0.9)]),
                         [Segment(10, 14, 0.9)])


class PostprocessClipTest(unittest.TestCase):

    def setUp(self):
        self.manifest = self._load()
        self.clip = self.manifest.clips[0]

    @staticmethod
    def _load():
        with tempfile.TemporaryDirectory() as folder:
            write_run_folder(folder)
            return load_manifest(os.path.join(folder, "manifest.json"))

    def test_ignores_unknown_and_fills_missing_tags(self):
        tags = postprocess_clip(MODEL_ANSWER, self.manifest, self.clip)
        self.assertEqual([t.name for t in tags], [WALK, IDLE])
        self.assertEqual(tags[0].segments, (Segment(10, 60, 0.9),))
        self.assertEqual(tags[0].reasoning, "It walks.")
        self.assertEqual(tags[1], TagResult(IDLE, "", ()))

    def test_skips_malformed_segments(self):
        answer = {"tags": [{"name": WALK, "reasoning": "r",
                            "segments": [{"startFrame": 20}, {"startFrame": 30.4,
                                                              "endFrame": 50,
                                                              "confidence": 0.8}]}]}
        tags = postprocess_clip(answer, self.manifest, self.clip)
        self.assertEqual(tags[0].segments, (Segment(30, 50, 0.8),))

    def test_rejects_response_without_tags(self):
        with self.assertRaises(ValueError):
            postprocess_clip({"answer": []}, self.manifest, self.clip)


class PromptTest(unittest.TestCase):

    def setUp(self):
        self.manifest = PostprocessClipTest._load()

    def test_prompt_lists_tags_context_and_counter(self):
        text = build_prompt(self.manifest, self.manifest.clips[0])
        for expected in (WALK, IDLE, "Character walks at a steady pace",
                         "What the character's body is doing", "`frame N`",
                         "frames 10 (inclusive) to 100 (exclusive)", "1.40 m/s", "EVERY",
                         "Several tags may apply to the same frames"):
            self.assertIn(expected, text)

    def test_schema_constrains_tag_names(self):
        schema = response_schema([WALK, IDLE])
        tag = schema["properties"]["tags"]["items"]
        self.assertEqual(tag["properties"]["name"]["enum"], [WALK, IDLE])
        segment = tag["properties"]["segments"]["items"]
        self.assertEqual(set(segment["required"]), {"startFrame", "endFrame", "confidence"})


class ManifestRoundTripTest(unittest.TestCase):

    def test_load_manifest_and_write_results(self):
        with tempfile.TemporaryDirectory() as folder:
            write_run_folder(folder)
            manifest = load_manifest(os.path.join(folder, "manifest.json"))
            clip = manifest.clips[0]
            self.assertEqual((clip.start_frame, clip.end_frame, clip.frame_step), (10, 100, 3))
            self.assertEqual(manifest.tags[0].ancestors[0].name, "animation.action")

            results = Results("m", (
                ClipResult(GUID, None, (TagResult(WALK, "r", (Segment(12, 40, 0.9),)),)),
                ClipResult("bad", "boom", ())))
            path = os.path.join(folder, "results.json")
            write_results(path, results)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data, {"model": "m", "clips": [
            {"guid": GUID, "error": None, "tags": [
                {"name": WALK, "reasoning": "r",
                 "segments": [{"startFrame": 12, "endFrame": 40, "confidence": 0.9}]}]},
            {"guid": "bad", "error": "boom", "tags": []}]})

    def test_malformed_manifest_raises_value_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"model": "m"}, f)
            with self.assertRaises(ValueError):
                load_manifest(path)


class AnnotateTest(unittest.TestCase):

    def _run(self, folder, **kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = annotate.run(folder, **kwargs)
        with open(os.path.join(folder, "results.json"), encoding="utf-8") as f:
            return code, out.getvalue().splitlines(), json.load(f)

    def test_resume_reparses_raw_response_without_calling_the_model(self):
        with tempfile.TemporaryDirectory() as folder:
            write_run_folder(folder)
            os.makedirs(os.path.join(folder, "raw"))
            with open(os.path.join(folder, "raw", f"{GUID}.json"), "w", encoding="utf-8") as f:
                json.dump({"responses": [{"text": json.dumps(MODEL_ANSWER), "inputTokens": 1,
                                          "outputTokens": 2}]}, f)

            code, lines, results = self._run(folder, resume=True)

        self.assertEqual(code, 0)
        self.assertEqual(lines, ["PROGRESS 0 1 walk_01", "DONE 1 0"])
        tags = results["clips"][0]["tags"]
        self.assertEqual([t["name"] for t in tags], [WALK, IDLE])
        self.assertEqual(tags[0]["segments"],
                         [{"startFrame": 10, "endFrame": 60, "confidence": 0.9}])

    def test_injected_call_saves_raw_and_reports_usage(self):
        calls = []

        def call_model(video, requests, model, resolution, fps):
            calls.append((video, model, resolution, fps))
            return [(json.dumps(MODEL_ANSWER), 1200, 80)]

        with tempfile.TemporaryDirectory() as folder:
            write_run_folder(folder)
            code, lines, _ = self._run(folder, model="other", call_model=call_model)
            with open(os.path.join(folder, "raw", f"{GUID}.json"), encoding="utf-8") as f:
                raw = json.load(f)

        self.assertEqual(code, 0)
        self.assertEqual(calls[0][1:], ("other", "low", 10.0))
        self.assertTrue(calls[0][0].endswith(f"{GUID}.mp4"))
        self.assertIn(f"USAGE {GUID} 1200 80", lines)
        self.assertEqual((raw["responses"][0]["inputTokens"], raw["responses"][0]["outputTokens"]),
                         (1200, 80))

    def test_per_tag_requests_ask_about_one_tag_each_and_merge(self):
        requests_seen = []

        def call_model(video, requests, model, resolution, fps):
            requests_seen.extend(requests)
            answers = []
            for _, schema in requests:
                name = schema["properties"]["tags"]["items"]["properties"]["name"]["enum"][0]
                own = [t for t in MODEL_ANSWER["tags"] if t["name"] == name]
                answers.append((json.dumps({"tags": own}), 100, 10))
            return answers

        with tempfile.TemporaryDirectory() as folder:
            write_run_folder(folder, perTagRequests=True)
            code, lines, results = self._run(folder, call_model=call_model)

        self.assertEqual(code, 0)
        self.assertEqual(len(requests_seen), 2)
        for (text, schema), own, other in zip(requests_seen, (WALK, IDLE), (IDLE, WALK)):
            self.assertIn(own, text)
            self.assertNotIn(other, text)
        self.assertIn(f"USAGE {GUID} 200 20", lines)
        tags = results["clips"][0]["tags"]
        self.assertEqual([t["name"] for t in tags], [WALK, IDLE])
        self.assertEqual(tags[0]["segments"],
                         [{"startFrame": 10, "endFrame": 60, "confidence": 0.9}])

    def test_failing_clip_is_recorded_and_exits_2(self):
        with tempfile.TemporaryDirectory() as folder:
            write_run_folder(folder)
            code, lines, results = self._run(
                folder, call_model=lambda *args: [("not json", 1, 1)])

        self.assertEqual(code, 2)
        self.assertTrue(any(line.startswith(f"CLIPERROR {GUID} ") for line in lines))
        self.assertEqual(lines[-1], "DONE 0 1")
        self.assertIsNotNone(results["clips"][0]["error"])

    def test_missing_api_key_is_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            write_run_folder(folder)
            saved = os.environ.pop(annotate.API_KEY_VARIABLE, None)
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = annotate.run(folder)
            finally:
                if saved is not None:
                    os.environ[annotate.API_KEY_VARIABLE] = saved
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(os.path.join(folder, "results.json")))

    def test_does_not_import_the_sdk(self):
        self.assertNotIn("autotag.gemini", sys.modules)
        self.assertNotIn("google.genai", sys.modules)


if __name__ == "__main__":
    unittest.main()
