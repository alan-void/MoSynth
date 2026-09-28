"""
The run folder's two JSON documents: the ``manifest.json`` Unity writes and the ``results.json``
it reads back.

Both use camelCase keys, because the C# side reads and writes them with the same names; the
dataclasses here use snake_case and translate at the boundary. Frame numbers are clip-local and
ranges are half-open, ``[startFrame, endFrame)``, throughout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TagAncestor:
    name: str
    description: str


@dataclass(frozen=True)
class TagSpec:
    name: str
    description: str
    ancestors: tuple[TagAncestor, ...]  # nearest parent first


@dataclass(frozen=True)
class ClipSpec:
    guid: str
    name: str
    asset_path: str
    clip_frame_rate: float
    start_frame: int
    end_frame: int
    render_fps: float
    frame_step: int
    video: str  # relative to the run folder
    ground_speed: tuple[float, ...]  # m/s, one per whole second of the slice


@dataclass(frozen=True)
class Manifest:
    model: str
    media_resolution: str
    min_confidence: float
    min_span_frames: int
    min_gap_frames: int
    per_tag_requests: bool  # one model request per tag instead of one per clip
    tags: tuple[TagSpec, ...]
    clips: tuple[ClipSpec, ...]


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int
    confidence: float


@dataclass(frozen=True)
class TagResult:
    name: str
    reasoning: str
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class ClipResult:
    guid: str
    error: Optional[str]
    tags: tuple[TagResult, ...]


@dataclass(frozen=True)
class Results:
    model: str
    clips: tuple[ClipResult, ...]


def _tag_from_json(entry: dict) -> TagSpec:
    return TagSpec(
        name=entry["name"],
        description=entry.get("description", ""),
        ancestors=tuple(TagAncestor(a["name"], a.get("description", ""))
                        for a in entry.get("ancestors", [])))


def _clip_from_json(entry: dict) -> ClipSpec:
    return ClipSpec(
        guid=entry["guid"],
        name=entry["name"],
        asset_path=entry.get("assetPath", ""),
        clip_frame_rate=float(entry["clipFrameRate"]),
        start_frame=int(entry["startFrame"]),
        end_frame=int(entry["endFrame"]),
        render_fps=float(entry["renderFps"]),
        frame_step=max(1, int(entry["frameStep"])),
        video=entry["video"],
        ground_speed=tuple(float(s) for s in entry.get("groundSpeed", [])))


def load_manifest(path: str) -> Manifest:
    """Reads ``manifest.json``; raises ``ValueError`` naming the problem when it is malformed."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        return Manifest(
            model=data["model"],
            media_resolution=data.get("mediaResolution", "low"),
            min_confidence=float(data["minConfidence"]),
            min_span_frames=int(data["minSpanFrames"]),
            min_gap_frames=int(data["minGapFrames"]),
            per_tag_requests=bool(data.get("perTagRequests", False)),
            tags=tuple(_tag_from_json(t) for t in data["tags"]),
            clips=tuple(_clip_from_json(c) for c in data["clips"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed manifest {path}: {e!r}") from e


def results_to_json(results: Results) -> dict:
    return {
        "model": results.model,
        "clips": [{
            "guid": clip.guid,
            "error": clip.error,
            "tags": [{
                "name": tag.name,
                "reasoning": tag.reasoning,
                "segments": [{"startFrame": s.start_frame,
                              "endFrame": s.end_frame,
                              "confidence": s.confidence} for s in tag.segments],
            } for tag in clip.tags],
        } for clip in results.clips],
    }


def write_results(path: str, results: Results) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_to_json(results), f, indent=2, ensure_ascii=False)
