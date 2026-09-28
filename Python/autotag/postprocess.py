"""
Turns a model's raw answer into the segments ``results.json`` promises.

The model reads frame numbers off a counter burned into a video sampled every few clip frames, so
its ranges are approximate: they can stray outside the slice, overlap, or leave a gap of a frame or
two inside what is really one span. This module is where those are reconciled, so that every
segment written is sorted, non-overlapping, non-adjacent, inside the slice, at least
``min_span_frames`` long and at least ``min_confidence`` sure. It is pure and stdlib-only.
"""

from __future__ import annotations

from typing import Any, Iterable

from autotag.manifest import ClipSpec, Manifest, Segment, TagResult


def clean_segments(segments: Iterable[Segment], start_frame: int, end_frame: int,
                   min_confidence: float, min_span_frames: int,
                   min_gap_frames: int) -> list[Segment]:
    """Filters, clamps and merges one tag's segments into the guaranteed form."""
    kept = []
    for s in segments:
        if s.confidence < min_confidence:
            continue
        start, end = max(s.start_frame, start_frame), min(s.end_frame, end_frame)
        if start < end:
            kept.append(Segment(start, end, min(s.confidence, 1.0)))
    kept.sort(key=lambda s: (s.start_frame, s.end_frame))

    # A gap of zero frames is still two touching spans, so merging always covers adjacency.
    merge_below = max(min_gap_frames, 1)
    merged: list[Segment] = []
    for s in kept:
        if merged and s.start_frame - merged[-1].end_frame < merge_below:
            last = merged[-1]
            merged[-1] = Segment(last.start_frame, max(last.end_frame, s.end_frame),
                                 max(last.confidence, s.confidence))
        else:
            merged.append(s)
    return [s for s in merged if s.end_frame - s.start_frame >= min_span_frames]


def _raw_segment(entry: Any) -> Segment | None:
    try:
        return Segment(int(round(float(entry["startFrame"]))),
                       int(round(float(entry["endFrame"]))),
                       float(entry["confidence"]))
    except (KeyError, TypeError, ValueError):
        return None


def postprocess_clip(response: Any, manifest: Manifest, clip: ClipSpec) -> tuple[TagResult, ...]:
    """
    One ``TagResult`` per manifest tag, in manifest order, from the parsed model response.

    Tags the model invented are ignored and tags it left out come back empty, so the result always
    lists exactly the manifest's tags. A malformed segment entry is skipped rather than failing
    the clip. Raises ``ValueError`` only when the response has no ``tags`` list at all.
    """
    if not isinstance(response, dict) or not isinstance(response.get("tags"), list):
        raise ValueError("response has no 'tags' list")

    names = [t.name for t in manifest.tags]
    reasoning: dict[str, str] = {}
    raw: dict[str, list[Segment]] = {name: [] for name in names}
    for entry in response["tags"]:
        if not isinstance(entry, dict) or entry.get("name") not in raw:
            continue
        name = entry["name"]
        if not reasoning.get(name):
            reasoning[name] = str(entry.get("reasoning") or "")
        segments = entry.get("segments")
        if not isinstance(segments, list):
            continue
        raw[name].extend(s for s in map(_raw_segment, segments) if s is not None)

    return tuple(
        TagResult(name, reasoning.get(name, ""), tuple(clean_segments(
            raw[name], clip.start_frame, clip.end_frame, manifest.min_confidence,
            manifest.min_span_frames, manifest.min_gap_frames)))
        for name in names)
