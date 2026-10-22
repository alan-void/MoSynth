"""
Tiles every method's benchmark video of one path into a single labelled grid video.

A sweep run with ``recordVideo`` on writes one MP4 per run into ``<sweep>/videos/``, each with a JSON
sidecar naming its method, path and seed, and the frame at which its settle time ended. This reads
those sidecars, groups the runs by path, and writes ``<sweep>/videos/compare_<path>.mp4``: one tile
per method in the sweep's method order, all starting at their first measured frame, so the tiles
are synchronised from the start of measurement rather than from spawn. A run that finished sooner
holds its last frame; a method with no video for a path gets a blank tile saying so.

Usage, from the ``Python/`` folder::

    python -m benchmark.compose_videos ../Benchmarks/showcase_20261021_120000
    python -m benchmark.compose_videos ../Benchmarks/showcase_20261021_120000 --stills 10 40 70 95

``--stills`` also saves PNG frames of each comparison at those percentages of its length into
``<sweep>/stills/``. Needs ``imageio-ffmpeg`` (``requirements-video.txt``), which brings its own
ffmpeg binary, and Pillow for the labels.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (40, 43, 48)
LABEL_BOX = (14, 16, 20)
LABEL_ALPHA = 0.72


@dataclass
class Run:
    """One run's video, as its sidecar describes it."""
    video: str
    method: str
    label: str
    path: str
    seeded: bool
    seed: int
    run_index: int
    frame_rate: float
    width: int
    height: int
    frame_count: int
    measure_start_frame: int

    @property
    def display_name(self) -> str:
        return f"{self.method}  (seed {self.seed})" if self.seeded else self.method

    @property
    def first_frame(self) -> int:
        """First frame to show: measurement start, or the last frame of a run that never got there."""
        if self.measure_start_frame >= 0:
            return min(self.measure_start_frame, max(0, self.frame_count - 1))
        return max(0, self.frame_count - 1)


def read_sidecars(videos_dir: str) -> list[Run]:
    runs = []
    for sidecar in sorted(glob.glob(os.path.join(videos_dir, "*.json"))):
        with open(sidecar, encoding="utf-8") as handle:
            data = json.load(handle)
        if "video" not in data or "path" not in data:
            continue
        runs.append(Run(
            video=os.path.join(videos_dir, data["video"]),
            method=data.get("method", data.get("label", "?")),
            label=data.get("label", data.get("method", "?")),
            path=data["path"],
            seeded=bool(data.get("seeded", False)),
            seed=int(data.get("seed", 0)),
            run_index=int(data.get("runIndex", 0)),
            frame_rate=float(data.get("frameRate", 30.0)),
            width=int(data.get("width", 1920)),
            height=int(data.get("height", 1080)),
            frame_count=int(data.get("frameCount", 0)),
            measure_start_frame=int(data.get("measureStartFrame", -1)),
        ))
    return runs


def grid_shape(count: int) -> tuple[int, int]:
    """Columns and rows of the most nearly square grid holding ``count`` tiles, wider than tall."""
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / columns))
    return columns, rows


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class Label:
    """A tile's caption, rendered once and alpha-blended onto every frame."""

    def __init__(self, text: str, tile_height: int):
        size = max(14, tile_height // 22)
        font = load_font(size)
        pad = size // 2
        left, top, right, bottom = font.getbbox(text)
        width = right - left + 2 * pad
        height = bottom - top + 2 * pad

        image = Image.new("RGB", (width, height), LABEL_BOX)
        ImageDraw.Draw(image).text((pad - left, pad - top), text, font=font, fill=(255, 255, 255))

        rgb = np.asarray(image, dtype=np.float32)
        # Text pixels opaque, the box behind them translucent.
        is_text = (rgb.max(axis=2, keepdims=True) > max(LABEL_BOX) + 8)
        self.alpha = np.where(is_text, 1.0, LABEL_ALPHA).astype(np.float32)
        self.rgb = rgb
        self.margin = pad

    def draw(self, tile: np.ndarray) -> None:
        # Bottom-left, because the top-left of every tile already holds the recorder's own overlay.
        x = self.margin
        y = max(0, tile.shape[0] - self.rgb.shape[0] - self.margin)
        h = min(self.rgb.shape[0], tile.shape[0] - y)
        w = min(self.rgb.shape[1], tile.shape[1] - x)
        if h <= 0 or w <= 0:
            return
        region = tile[y:y + h, x:x + w].astype(np.float32)
        alpha = self.alpha[:h, :w]
        tile[y:y + h, x:x + w] = (region * (1.0 - alpha) + self.rgb[:h, :w] * alpha).astype(np.uint8)


class TileSource:
    """Frames of one run, scaled to the tile and starting at measurement; repeats its last frame once spent."""

    def __init__(self, run: Run | None, width: int, height: int, caption: str):
        self.width, self.height = width, height
        self.label = Label(caption, height)
        self.last = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
        self.length = 0
        self._frames = None

        if run is None or not os.path.isfile(run.video):
            self._write_missing(caption)
            return

        self.length = max(0, run.frame_count - run.first_frame)
        self._frames = imageio_ffmpeg.read_frames(
            run.video, output_params=["-vf", f"scale={width}:{height}:flags=area"])
        next(self._frames)  # metadata
        for _ in range(run.first_frame):
            if not self._advance():
                break

    def _write_missing(self, caption: str) -> None:
        image = Image.fromarray(self.last)
        font = load_font(max(14, self.height // 18))
        text = "no video"
        left, top, right, bottom = font.getbbox(text)
        position = ((self.width - (right - left)) // 2, (self.height - (bottom - top)) // 2)
        ImageDraw.Draw(image).text(position, text, font=font, fill=(150, 155, 160))
        self.last = np.asarray(image).copy()

    def _advance(self) -> bool:
        if self._frames is None:
            return False
        try:
            raw = next(self._frames)
        except StopIteration:
            self._frames = None
            return False
        frame = np.frombuffer(raw, dtype=np.uint8)
        if frame.size != self.width * self.height * 3:
            self._frames = None
            return False
        self.last = frame.reshape(self.height, self.width, 3).copy()
        return True

    def next_frame(self) -> np.ndarray:
        self._advance()
        tile = self.last.copy()
        self.label.draw(tile)
        return tile

    def close(self) -> None:
        if self._frames is not None:
            self._frames.close()
            self._frames = None


def compose_path(path: str, methods: list[tuple[int, str]], runs: dict[tuple[str, str], Run],
                 out_file: str, tile_width: int, stills: list[float], stills_dir: str) -> int:
    """Writes one comparison video; returns its frame count."""
    present = [run for (_, label) in methods if (run := runs.get((label, path))) is not None]
    first = present[0]
    aspect = first.width / max(1, first.height)
    tile_w = tile_width - tile_width % 2
    tile_h = int(round(tile_w / aspect))
    tile_h -= tile_h % 2
    columns, rows = grid_shape(len(methods))
    size = (columns * tile_w, rows * tile_h)

    sources = []
    for _, label in methods:
        run = runs.get((label, path))
        caption = run.display_name if run is not None else label
        sources.append(TileSource(run, tile_w, tile_h, caption))

    length = max(1, max(source.length for source in sources))
    still_frames = {min(length - 1, int(round(p / 100.0 * (length - 1)))): p for p in stills}

    writer = imageio_ffmpeg.write_frames(
        out_file, size, fps=first.frame_rate, codec="libx264", quality=None, macro_block_size=2,
        output_params=["-crf", "18", "-preset", "medium"])
    writer.send(None)
    canvas = np.full((size[1], size[0], 3), BACKGROUND, dtype=np.uint8)
    try:
        for index in range(length):
            for slot, source in enumerate(sources):
                row, column = divmod(slot, columns)
                canvas[row * tile_h:(row + 1) * tile_h, column * tile_w:(column + 1) * tile_w] = source.next_frame()
            writer.send(canvas.tobytes())
            if index in still_frames:
                os.makedirs(stills_dir, exist_ok=True)
                name = f"compare_{safe_name(path)}_{int(round(still_frames[index])):02d}.png"
                Image.fromarray(canvas).save(os.path.join(stills_dir, name))
    finally:
        writer.close()
        for source in sources:
            source.close()

    return length


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweep", help="Sweep folder holding videos/ with the per-run MP4s and sidecars.")
    parser.add_argument("--tile-width", type=int, default=960,
                        help="Width of each tile in pixels; 960 makes a 2x2 grid 1920x1080.")
    parser.add_argument("--stills", type=float, nargs="*", default=[],
                        help="Also save PNG frames at these percentages of each comparison into <sweep>/stills/.")
    args = parser.parse_args(argv)

    # Scaling on read is deliberate; imageio-ffmpeg warns about it once per video otherwise.
    logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

    videos_dir = os.path.join(args.sweep, "videos")
    runs = read_sidecars(videos_dir)
    if not runs:
        print(f"No video sidecars in {videos_dir}; was the sweep run with recordVideo on?")
        return 1

    # Every method of the sweep gets a tile on every path, so a missing run reads as missing.
    methods = sorted({(run.run_index, run.label) for run in runs})
    by_key = {(run.label, run.path): run for run in runs}
    paths = list(dict.fromkeys(run.path for run in sorted(runs, key=lambda r: (r.path, r.run_index))))

    for path in paths:
        out_file = os.path.join(videos_dir, f"compare_{safe_name(path)}.mp4")
        frames = compose_path(path, methods, by_key, out_file, args.tile_width, args.stills,
                              os.path.join(args.sweep, "stills"))
        print(f"{out_file}: {len(methods)} tile(s), {frames} frame(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
