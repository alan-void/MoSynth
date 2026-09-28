"""
The instructions and the response schema sent with each clip's video.

The model answers in clip frame numbers it reads off the counter burned into every video frame,
not in video time, so its answer lines up with the clip however the video was sampled.
"""

from __future__ import annotations

from typing import Optional, Sequence

from autotag.manifest import ClipSpec, Manifest, TagSpec


def _describe_tag(tag: TagSpec) -> str:
    lines = [f"- `{tag.name}`: {tag.description or '(no description)'}"]
    for ancestor in tag.ancestors:
        lines.append(f"    - within `{ancestor.name}`: {ancestor.description}")
    return "\n".join(lines)


def _describe_ground_speed(clip: ClipSpec) -> str:
    if not clip.ground_speed:
        return "Not available."
    lines = []
    for second, speed in enumerate(clip.ground_speed):
        first = clip.start_frame + round(second * clip.clip_frame_rate)
        last = min(clip.start_frame + round((second + 1) * clip.clip_frame_rate), clip.end_frame)
        lines.append(f"- frames {first}-{last - 1}: {speed:.2f} m/s")
    return "\n".join(lines)


def build_prompt(manifest: Manifest, clip: ClipSpec,
                 tags: Optional[Sequence[TagSpec]] = None) -> str:
    """The instructions for ``tags`` (every manifest tag by default) on one clip's video."""
    tags = "\n".join(_describe_tag(t) for t in (manifest.tags if tags is None else tags))
    return f"""\
You are annotating a motion-capture animation clip with gameplay tags.

## What the video shows
A character drawn as a capsule stick figure: left limbs are blue, right limbs are red, the spine
and head are grey. It moves over a checkered ground plane. The camera follows the character but
never rotates, so turning shows as the figure rotating on screen, and travel shows as the checker
pattern sliding past.

A counter reading `frame N` in the top-left corner of every video frame gives the clip frame
number that video frame shows. The video is sampled every {clip.frame_step} clip frame(s), at
{clip.render_fps:g} video frames per second, so consecutive counters step by {clip.frame_step}.
Always read frame numbers from that counter; never compute them from video time.

The clip covers frames {clip.start_frame} (inclusive) to {clip.end_frame} (exclusive), at
{clip.clip_frame_rate:g} clip frames per second.

## Tags
Each tag is listed with its description, followed by the descriptions of the broader categories it
belongs to, as context.

{tags}

## Supplementary evidence
Measured mean ground speed of the character per second of the clip:
{_describe_ground_speed(clip)}

## What to report
The tags are independent, not alternatives: judge each one on its own description, as if it were
the only tag asked about. Several tags may apply to the same frames (a character can be running and
strafing at once), and a stretch may have no tag at all.

Report EVERY tag listed above, exactly once, by its full name. For each tag give:
- `reasoning`: one sentence on what you saw.
- `segments`: every stretch of the clip during which the tag applies, as half-open frame ranges
  [startFrame, endFrame) in clip frame numbers read from the counter, each with a `confidence`
  between 0 and 1. Use an empty list if the tag never applies.
"""


def response_schema(tag_names: Sequence[str]) -> dict:
    """JSON schema for the answer; ``reasoning`` precedes ``segments`` so the model reasons first."""
    segment = {
        "type": "object",
        "properties": {
            "startFrame": {"type": "integer"},
            "endFrame": {"type": "integer"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["startFrame", "endFrame", "confidence"],
    }
    tag = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": list(tag_names)},
            "reasoning": {"type": "string"},
            "segments": {"type": "array", "items": segment},
        },
        "required": ["name", "reasoning", "segments"],
    }
    return {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": tag}},
        "required": ["tags"],
    }
