"""
Annotates every clip of a run folder Unity rendered, and writes ``results.json`` for Unity to apply.

Each clip's video goes to the model with the manifest's tags; the answer is stored verbatim in
``raw/<guid>.json`` before it is parsed, so a run that failed while parsing or post-processing can
be repeated with ``--resume`` without paying for the calls again. A clip that fails is recorded
with its error and the run moves on.

Run from the ``Python/`` folder, with the key in ``GEMINI_API_KEY``::

    python -m autotag.annotate ../Library/AutoTag/20260923-101500
    python -m autotag.annotate <run-folder> --model gemini-3.8-flash --only <guid> <guid>
    python -m autotag.annotate <run-folder> --resume

Unity parses these stdout lines; anything else is only logged::

    PROGRESS <index> <total> <clipName>
    USAGE <guid> <inputTokens> <outputTokens>
    CLIPERROR <guid> <message>
    DONE <succeeded> <failed>

Exit code: 0 when every clip succeeded, 2 when some failed, 1 when the run could not start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Optional, Sequence

from autotag import postprocess, prompt
from autotag.manifest import ClipResult, ClipSpec, Manifest, Results, load_manifest, write_results

API_KEY_VARIABLE = "GEMINI_API_KEY"

# (video_path, [(prompt, schema)], model, media_resolution, fps)
#   -> [(text, input_tokens, output_tokens)], one per request, in order
CallModel = Callable[[str, Sequence[tuple[str, dict]], str, str, float],
                     list[tuple[str, int, int]]]


def _raw_path(run_folder: str, guid: str) -> str:
    return os.path.join(run_folder, "raw", f"{guid}.json")


def _single_line(error: BaseException) -> str:
    return " ".join(f"{type(error).__name__}: {error}".split())


def _requests(manifest: Manifest, clip: ClipSpec) -> list[tuple[str, dict]]:
    """One request covering every tag, or one per tag when the manifest asks for that."""
    groups = [(t,) for t in manifest.tags] if manifest.per_tag_requests else [manifest.tags]
    return [(prompt.build_prompt(manifest, clip, group),
             prompt.response_schema([t.name for t in group])) for group in groups]


def _response_texts(run_folder: str, manifest: Manifest, clip: ClipSpec, model: str,
                    resume: bool, call_model: Optional[CallModel]) -> list[str]:
    raw_path = _raw_path(run_folder, clip.guid)
    if resume and os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
        return [r["text"] for r in raw["responses"]]

    responses = call_model(os.path.join(run_folder, clip.video), _requests(manifest, clip),
                           model, manifest.media_resolution, clip.render_fps)
    input_tokens = sum(r[1] for r in responses)
    output_tokens = sum(r[2] for r in responses)
    print(f"USAGE {clip.guid} {input_tokens} {output_tokens}", flush=True)

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "responses": [
            {"text": text, "inputTokens": i, "outputTokens": o} for text, i, o in responses]},
            f, indent=2, ensure_ascii=False)
    return [text for text, _, _ in responses]


def _merge_responses(texts: Sequence[str]) -> dict:
    """Joins the ``tags`` lists of several responses into one response."""
    merged = []
    for text in texts:
        response = json.loads(text)
        if not isinstance(response, dict) or not isinstance(response.get("tags"), list):
            raise ValueError("response has no 'tags' list")
        merged.extend(response["tags"])
    return {"tags": merged}


def _annotate_clip(run_folder: str, manifest: Manifest, clip: ClipSpec, model: str,
                   resume: bool, call_model: Optional[CallModel]) -> ClipResult:
    try:
        texts = _response_texts(run_folder, manifest, clip, model, resume, call_model)
        tags = postprocess.postprocess_clip(_merge_responses(texts), manifest, clip)
        return ClipResult(clip.guid, None, tags)
    except Exception as e:
        message = _single_line(e)
        print(f"CLIPERROR {clip.guid} {message}", flush=True)
        return ClipResult(clip.guid, message, ())


def run(run_folder: str, model: Optional[str] = None, only: Optional[Sequence[str]] = None,
        resume: bool = False, call_model: Optional[CallModel] = None) -> int:
    """Annotates the run folder and returns the process exit code."""
    try:
        manifest = load_manifest(os.path.join(run_folder, "manifest.json"))
    except (OSError, ValueError) as e:
        print(f"error: cannot read the manifest: {_single_line(e)}", file=sys.stderr, flush=True)
        return 1
    model = model or manifest.model

    clips = manifest.clips
    if only:
        wanted = set(only)
        for guid in wanted - {c.guid for c in clips}:
            print(f"warning: --only {guid} is not in the manifest", flush=True)
        clips = tuple(c for c in clips if c.guid in wanted)

    needs_api = any(not (resume and os.path.exists(_raw_path(run_folder, c.guid))) for c in clips)
    if needs_api and call_model is None:
        if not os.environ.get(API_KEY_VARIABLE):
            print(f"error: set {API_KEY_VARIABLE} to a Gemini API key", file=sys.stderr, flush=True)
            return 1
        try:
            from autotag import gemini
        except ImportError as e:
            print(f"error: the google-genai SDK is unavailable: {e}", file=sys.stderr, flush=True)
            return 1
        call_model = gemini.annotate_video

    results = []
    for index, clip in enumerate(clips):
        print(f"PROGRESS {index} {len(clips)} {clip.name}", flush=True)
        results.append(_annotate_clip(run_folder, manifest, clip, model, resume, call_model))

    write_results(os.path.join(run_folder, "results.json"), Results(model, tuple(results)))
    failed = sum(1 for r in results if r.error is not None)
    print(f"DONE {len(results) - failed} {failed}", flush=True)
    return 2 if failed else 0


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str):
        # argparse exits with 2 by default, which the protocol reserves for failed clips.
        self.print_usage(sys.stderr)
        self.exit(1, f"error: {message}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("run_folder")
    parser.add_argument("--model", help="overrides the manifest's model")
    parser.add_argument("--only", nargs="+", metavar="GUID", help="annotate only these clips")
    parser.add_argument("--resume", action="store_true",
                        help="re-parse raw/<guid>.json where it exists instead of calling the API")
    args = parser.parse_args(argv)
    return run(args.run_folder, args.model, args.only, args.resume)


if __name__ == "__main__":
    sys.exit(main())
