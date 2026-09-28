"""
The one module that talks to Gemini, through the ``google-genai`` Interactions API.

Nothing but :mod:`autotag.annotate` imports it, and that lazily, so the rest of the package and
its tests run without the SDK installed. The API key is read from ``GEMINI_API_KEY`` by the client.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence, TypeVar

import httpx
from google import genai
# The Interactions API's connection error has no public import path in google-genai 2.x.
from google.genai._gaos.lib.compat_errors import APIConnectionError

ATTEMPTS = 3
FIRST_BACKOFF_SECONDS = 2.0
UPLOAD_POLL_SECONDS = 2.0
UPLOAD_TIMEOUT_SECONDS = 600.0
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}

T = TypeVar("T")


def _is_transient(error: BaseException) -> bool:
    # The Interactions API raises its own error hierarchy rather than google.genai.errors, so
    # this reads the HTTP status off either, and walks the chain to a wrapped transport error.
    while error is not None:
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        if status in TRANSIENT_STATUS_CODES:
            return True
        if isinstance(error, (APIConnectionError, httpx.TransportError,
                              ConnectionError, TimeoutError)):
            return True
        error = error.__cause__
    return False


def _with_retry(action: Callable[[], T]) -> T:
    for attempt in range(ATTEMPTS):
        try:
            return action()
        except Exception as e:
            if attempt == ATTEMPTS - 1 or not _is_transient(e):
                raise
            time.sleep(FIRST_BACKOFF_SECONDS * 2 ** attempt)
    raise AssertionError("unreachable")


def _wait_until_active(client: genai.Client, name: str):
    deadline = time.monotonic() + UPLOAD_TIMEOUT_SECONDS
    while True:
        uploaded = _with_retry(lambda: client.files.get(name=name))
        state = uploaded.state.name if uploaded.state is not None else "STATE_UNSPECIFIED"
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise RuntimeError(f"video processing failed: {uploaded.error}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"video still {state} after {UPLOAD_TIMEOUT_SECONDS:.0f}s")
        time.sleep(UPLOAD_POLL_SECONDS)


def _ask(client: genai.Client, uploaded, prompt: str, schema: dict, model: str,
         media_resolution: str, fps: float) -> tuple[str, int, int]:
    interaction = _with_retry(lambda: client.interactions.create(
        model=model,
        input=[
            {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type,
             "resolution": media_resolution,
             "processing": {"type": "static", "fps": fps}},
            {"type": "text", "text": prompt},
        ],
        response_format={"type": "text", "mime_type": "application/json", "schema": schema}))

    usage = interaction.usage
    input_tokens = (usage.total_input_tokens or 0) if usage else 0
    # Thinking tokens are billed as output but counted separately from the response's own.
    output_tokens = ((usage.total_output_tokens or 0) + (usage.total_thought_tokens or 0)
                     if usage else 0)
    return interaction.output_text or "", input_tokens, output_tokens


def annotate_video(video_path: str, requests: Sequence[tuple[str, dict]], model: str,
                   media_resolution: str, fps: float) -> list[tuple[str, int, int]]:
    """
    Uploads the video once, asks ``model`` each ``(prompt, schema)`` request about it, and deletes
    the upload. Returns each response's text and input and output token counts, in order.
    """
    client = genai.Client()
    uploaded = _with_retry(lambda: client.files.upload(file=video_path))
    try:
        uploaded = _wait_until_active(client, uploaded.name)
        return [_ask(client, uploaded, prompt, schema, model, media_resolution, fps)
                for prompt, schema in requests]
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as e:
            print(f"warning: could not delete uploaded file {uploaded.name}: {e}", flush=True)
