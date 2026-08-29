"""
Gait phase, derived from the foot contacts already stored in a pose database.

A phase-functioned network does not take a frame of animation and ask what to do next; it
takes *where in the gait cycle* the character is and blends a different set of weights for
each point on that cycle. That scalar has to come from somewhere, and the honest source is
the footfalls: a walk cycle is defined by them.

The construction here is the standard one.

* A **footfall** is a rising edge of a foot's contact flag -- the frame the foot goes down.
* Phase advances by pi per footfall, so a full stride, right heel to right heel, is 2*pi.
  Right footfalls land on 0 and left footfalls on pi.
* Between two footfalls phase is linear in frame index. Nothing measures how far through
  the swing the foot actually is, so anything else would be invention.
* Two footfalls of the *same* foot in a row -- the other foot's contact was missed, or the
  motion is a hop -- advance by 2*pi rather than pi, which keeps each foot on its own
  half of the cycle instead of swapping them for the rest of the clip.

Phase is unwrapped while it is being built, so its rate is a plain finite difference, and
only wrapped into [0, 2*pi) at the end.

Two cases have no cycle to measure and are reported as such rather than guessed at: a clip
with fewer than two footfalls (standing, or a fragment shorter than one step) gets phase 0
throughout and a rate of 0. A caller training on phase should drop those frames, which is
what ``phase_rate == 0`` marks.

Contacts come from ``PoseSet.foot_contacts``, which Unity fills by thresholding toe speed
and then median-filters -- so the edges here are already free of single-frame chatter.
"""

from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi

LEFT = 0
RIGHT = 1


def footfall_frames(contacts: np.ndarray) -> np.ndarray:
    """
    Frames where a contact flag rises, for one foot.

    A foot already in contact on frame 0 is not a footfall: the clip started mid-stance and
    nothing says when that foot went down.

    :param contacts: (n_frames,) boolean or 0/1 contact flags for one foot.
    :return: (n_events,) int64 frame indices.
    """
    flags = np.asarray(contacts).astype(bool)
    if flags.size < 2:
        return np.zeros(0, dtype=np.int64)

    return np.flatnonzero(flags[1:] & ~flags[:-1]).astype(np.int64) + 1


def _footfall_targets(events: np.ndarray, feet: np.ndarray) -> np.ndarray:
    """
    The unwrapped phase each footfall lands on: 0 for the first, then +pi per alternation
    and +2*pi when the same foot falls twice running.
    """
    targets = np.zeros(events.size, dtype=np.float64)
    for k in range(1, events.size):
        targets[k] = targets[k - 1] + (np.pi if feet[k] != feet[k - 1] else TAU)

    # The cycle is conventionally zeroed on a right footfall, so shift the whole clip by
    # half a cycle when it opens on a left one.
    if events.size > 0 and feet[0] == LEFT:
        targets += np.pi

    return targets


def _interpolate(n_frames: int, events: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Unwrapped phase per frame: linear between footfalls, and continuing at the adjacent
    segment's rate before the first and after the last, so a clip has no flat ends.
    """
    phase = np.interp(np.arange(n_frames, dtype=np.float64), events.astype(np.float64), targets)

    lead_rate = (targets[1] - targets[0]) / (events[1] - events[0])
    before = np.arange(n_frames) < events[0]
    phase[before] = targets[0] + (np.arange(n_frames)[before] - events[0]) * lead_rate

    trail_rate = (targets[-1] - targets[-2]) / (events[-1] - events[-2])
    after = np.arange(n_frames) > events[-1]
    phase[after] = targets[-1] + (np.arange(n_frames)[after] - events[-1]) * trail_rate

    return phase


def unwrapped_clip_phase(foot_contacts: np.ndarray) -> np.ndarray:
    """
    Unwrapped gait phase for one clip, in radians, increasing with time.

    :param foot_contacts: (n_frames, 2) contact flags, left in column 0 and right in
        column 1 -- the layout ``PoseSet.foot_contacts`` uses.
    :return: (n_frames,) float64. All zero when the clip holds fewer than two footfalls.
    """
    contacts = np.asarray(foot_contacts)
    n_frames = contacts.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)

    left = footfall_frames(contacts[:, LEFT])
    right = footfall_frames(contacts[:, RIGHT])

    events = np.concatenate([left, right])
    feet = np.concatenate([np.full(left.size, LEFT), np.full(right.size, RIGHT)])

    order = np.argsort(events, kind='stable')
    events = events[order]
    feet = feet[order]

    if events.size < 2:
        return np.zeros(n_frames, dtype=np.float64)

    return _interpolate(n_frames, events, _footfall_targets(events, feet))


def clip_phase(foot_contacts: np.ndarray, frame_time: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapped gait phase and its rate for one clip.

    :param foot_contacts: (n_frames, 2) contact flags, left then right.
    :param frame_time: seconds per frame, which turns the per-frame slope into rad/s.
    :return: ``(phase, phase_rate)``, both (n_frames,) float32. Phase is in [0, 2*pi);
        the rate is the derivative of the *unwrapped* phase, so it never shows the wrap as
        a spike, and is 0 exactly where the clip had no measurable cycle.
    """
    unwrapped = unwrapped_clip_phase(foot_contacts)
    if unwrapped.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    rate = np.gradient(unwrapped) / float(frame_time) if unwrapped.size > 1 \
        else np.zeros(1, dtype=np.float64)

    return (np.mod(unwrapped, TAU).astype(np.float32), rate.astype(np.float32))


def pose_set_phase(foot_contacts: np.ndarray, clip_ranges, frame_time: float) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Gait phase and its rate for a whole database, clip by clip.

    Phase is per clip and never carries across a boundary: two clips are two different
    recordings, and interpolating a stride between the end of one and the start of the next
    would invent a footfall that never happened.

    :param foot_contacts: (n_frames, 2) contact flags for the whole database.
    :param clip_ranges: iterable of ``(start, end)`` half-open frame ranges.
    :param frame_time: seconds per frame.
    :return: ``(phase, phase_rate)``, both (n_frames,) float32.
    """
    contacts = np.asarray(foot_contacts)
    n_frames = contacts.shape[0]
    phase = np.zeros(n_frames, dtype=np.float32)
    phase_rate = np.zeros(n_frames, dtype=np.float32)

    for start, end in clip_ranges:
        start, end = int(start), min(int(end), n_frames)
        if end <= start:
            continue
        phase[start:end], phase_rate[start:end] = clip_phase(contacts[start:end], frame_time)

    return phase, phase_rate
