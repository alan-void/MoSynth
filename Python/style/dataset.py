"""
Turns a labelled 100STYLE database into the pairs, inputs and targets the style network trains on.

The database is one ``.mmpose`` holding every style's clips, plus a ``<name>.style_labels.json``
sidecar that says which style and gait each clip is -- the ``.mmpose`` clip table carries only
frame ranges. Everything here works in the character frame of :mod:`training.training_data`.

**Pairs.** 100STYLE has no frame-paired data, so each styled frame is paired with the ``Neutral``
frame of the same gait nearest in *baked* gait phase, near-ties broken by the nearest planar
character-frame velocity. Velocity rather than speed, so a side-step to the left is never paired
with one to the right. Neutral frames are also paired with themselves, which is what makes
``Neutral`` the identity style rather than a special case. Transition takes (``TR*``) have no
neutral counterpart and are held out.

**Inputs** use the *causal* phase of :mod:`style.phase`, computed from the contacts, not the baked
phase: the network has to train on the phase it will see at runtime, extrapolation lag included.

**Targets** split the delta into a closed-form posture term and a learned dynamics term::

    R_out = R_in * P(x) * D(x)

``P(x)`` blends the per-(style, gait) mean delta by soft gait weights computed from the input, and
the dynamics target is whatever is left, ``P(x)^-1 * R_in^-1 * R_target``. Runtime returns the same
soft-weighted ``P(x)``, so the network learns exactly the remainder it will be composed with.

Conventions are :mod:`training.training_data`'s: quaternions xyzw, y-up left-handed, the character
facing +z, every rate per second.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from core.quaternions import blend_quaternions
from lmm.fk import parents_within
from pfnn.dataset import select_bones
from style.phase import causal_phase
from training.training_data import TrainingSet, rotations_to_6d

# Gait classes in a fixed order. Transitions are recognised separately and never paired.
GAITS = ('FW', 'BW', 'SW', 'FR', 'BR', 'SR', 'ID')
TRANSITION_PREFIX = 'TR'
IDLE_GAIT = 'ID'

# The root velocity's components in the character frame. The frame faces +z (y-up, left-handed),
# so z is forward and x is lateral, positive to the character's right. See core.simulation_frame.
FORWARD_AXIS = 2
LATERAL_AXIS = 0

# The descriptor's leading entries; contact-bone heights follow.
DESCRIPTOR_FIXED = ('forward_velocity', 'lateral_velocity', 'yaw_rate')

# Phase pairing: candidates taken from the k-d tree per styled frame, and how far beyond the best
# phase match (chord distance on the unit circle, ~radians) a candidate still counts as a tie.
PAIRING_CANDIDATES = 32
PHASE_TIE_TOLERANCE = 0.15

# Cadence filter. A gait is measured only when its neutral clips step at least this often.
MIN_MEASURABLE_FOOTFALL_RATE = 0.5
CADENCE_RATIO_RANGE = (0.6, 1.6)
# Loose, because a slow style plants longer at the same cadence (Old reads 0.80 against
# Neutral's 0.58); a hop or skip shows in the footfall rate and repeated feet instead.
DUTY_FACTOR_TOLERANCE = 0.3
REPEATED_FOOT_TOLERANCE = 0.25

# Speed ratio guards: below this neutral speed a ratio means nothing, so it stays 1.
MIN_RATIO_SPEED = 0.2
SPEED_RATIO_RANGE = (0.25, 4.0)

IDENTITY_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


# Labels ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ClipLabel:
    """
    One clip's style and gait.

    :param style: the style tag name exactly as the sidecar spells it. Kept verbatim so that the
        names a checkpoint reports back are the names Unity wrote.
    :param gait: the gait code, e.g. ``FW`` or ``TR2``.
    """

    style: str
    gait: str

    @property
    def is_transition(self) -> bool:
        return self.gait.startswith(TRANSITION_PREFIX)


def tag_leaf(name: str) -> str:
    """The last component of a dotted tag name: ``style.Old`` and ``Old`` both give ``Old``."""
    return str(name).strip().split('.')[-1]


def style_key(name: str) -> str:
    """How two style names are compared: by leaf, ignoring case."""
    return tag_leaf(name).casefold()


def labels_path(data_dir: str, db_name: str) -> str:
    return os.path.join(data_dir, f'{db_name}.style_labels.json')


def parse_labels(document: dict, clip_count: int) -> list[ClipLabel]:
    """
    Validate a sidecar document and read its entries.

    :param clip_count: how many clips the ``.mmpose`` holds, in its clip table.
    :raises ValueError: a malformed document, an unknown gait, or a count that differs from the
        database's -- the sidecar is then stale, and pairing by position would mislabel clips.
    """
    clips = document.get('clips') if isinstance(document, dict) else None
    if not isinstance(clips, list):
        raise ValueError('a style labels file must be {"clips": [{"style": ..., "gait": ...}]}')
    if len(clips) != clip_count:
        raise ValueError(f'the style labels name {len(clips)} clips but the database holds '
                         f'{clip_count}. Bake the database and its labels again.')

    labels = []
    for i, entry in enumerate(clips):
        if not isinstance(entry, dict) or 'style' not in entry or 'gait' not in entry:
            raise ValueError(f'style label {i} needs a "style" and a "gait"')
        style = str(entry['style']).strip()
        gait = tag_leaf(entry['gait']).upper()
        if not style:
            raise ValueError(f'style label {i} has an empty style')
        if gait not in GAITS and not gait.startswith(TRANSITION_PREFIX):
            raise ValueError(f'style label {i} has gait {entry["gait"]!r}; expected one of '
                             f'{", ".join(GAITS)} or a {TRANSITION_PREFIX}<n> transition')
        labels.append(ClipLabel(style, gait))
    return labels


def load_labels(data_dir: str, db_name: str, clip_count: int) -> list[ClipLabel]:
    """Read ``<db_name>.style_labels.json`` beside the database; see :func:`parse_labels`."""
    path = labels_path(data_dir, db_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'no style labels at {path}. Bake the config to write them.')
    with open(path, 'r', encoding='utf-8') as f:
        return parse_labels(json.load(f), clip_count)


def labels_for_stored_clips(labels: list[ClipLabel], clips: list, n_frames: int) \
        -> list[ClipLabel]:
    """
    The labels of the clips :func:`core.simulation_frame.clip_ranges` keeps, in order.

    That function drops empty clips, so a training set's ``clip_ranges`` can be shorter than the
    ``.mmpose`` clip table the labels are counted against.
    """
    return [label for label, clip in zip(labels, clips)
            if min(int(clip['end']), n_frames) > int(clip['start'])]


# Packing description ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StyleSpec:
    """
    What the packing depends on, stored in the checkpoint so the runtime packs identically.

    :param bone_names: every bone of the skeleton, in skeleton order.
    :param parents: (n_bones,) parent per bone, -1 for the root.
    :param styled_bones: (Bs,) indices of the styled bones, ascending; bone 0 first.
    :param contact_bones: (C,) bone index behind each contact slot, slot order.
    :param history_offsets: seconds into the past of each history descriptor, ascending.
    """

    bone_names: tuple
    parents: np.ndarray
    styled_bones: np.ndarray
    contact_bones: np.ndarray
    contact_bone_names: tuple
    frame_time: float
    history_offsets: tuple

    @property
    def styled_bone_names(self) -> list:
        return [self.bone_names[i] for i in self.styled_bones]

    @property
    def n_styled(self) -> int:
        return int(self.styled_bones.size)

    @property
    def n_contacts(self) -> int:
        return int(self.contact_bones.size)

    @property
    def descriptor_size(self) -> int:
        return len(DESCRIPTOR_FIXED) + self.n_contacts

    @property
    def n_history(self) -> int:
        return len(self.history_offsets)

    @property
    def history_frames(self) -> np.ndarray:
        """Each history offset as a whole number of frames back, at least one."""
        return np.maximum(1, np.rint(np.asarray(self.history_offsets, dtype=np.float64)
                                     / self.frame_time)).astype(np.int64)

    @property
    def input_size(self) -> int:
        return 6 * self.n_styled + 1 + self.descriptor_size + 3 \
            + self.n_history * self.descriptor_size

    @property
    def output_size(self) -> int:
        return 6 * self.n_styled + 1 + 4

    @property
    def styled_parents(self) -> np.ndarray:
        return parents_within(self.parents, self.styled_bones)


def build_spec(training_set: TrainingSet, excluded_bones=(), history_offsets=(0.1, 0.2)) \
        -> StyleSpec:
    """
    :raises ValueError: an excluded-bone list :func:`pfnn.dataset.select_bones` refuses, contact
        bones the skeleton lacks, or a contact count other than two -- the causal phase reads a
        left and a right foot.
    """
    names = training_set.contact_bone_names
    if names is None:
        raise ValueError('the database does not name its contact bones; regenerate it')
    if len(names) != 2:
        raise ValueError(f'the style stage reads a left and a right contact, not {len(names)}')
    missing = [name for name in names if name not in training_set.bone_names]
    if missing:
        raise ValueError(f'contact bones not in the skeleton: {", ".join(missing)}')

    offsets = tuple(sorted(float(offset) for offset in history_offsets))
    if any(offset <= 0.0 for offset in offsets):
        raise ValueError(f'history offsets are seconds into the past and must be > 0: {offsets}')

    parents = np.asarray(training_set.parents, dtype=np.int64)
    return StyleSpec(
        bone_names=tuple(training_set.bone_names),
        parents=parents,
        styled_bones=select_bones(training_set.bone_names, parents, excluded_bones),
        contact_bones=np.array([training_set.bone_names.index(name) for name in names],
                               dtype=np.int64),
        contact_bone_names=tuple(names),
        frame_time=float(training_set.frame_time),
        history_offsets=offsets)


# Per-frame quantities, shared by the trainer and the runtime -----------------------------------

def descriptor(positions: np.ndarray, root_velocity: np.ndarray, root_yaw_rate: np.ndarray,
               contact_bones: np.ndarray) -> np.ndarray:
    """
    ``[forward velocity, lateral velocity, yaw rate, contact-bone heights...]`` per frame.

    :param positions: (n, n_bones, 3) character-frame joint positions.
    :param root_velocity: (n, 3) the frame's own velocity in its own space; z is forward and x
        lateral (see :data:`FORWARD_AXIS`).
    :param root_yaw_rate: (n,) rad/s.
    :return: (n, 3 + C) float32.
    """
    positions = np.asarray(positions, dtype=np.float32)
    root_velocity = np.asarray(root_velocity, dtype=np.float32).reshape(-1, 3)
    return np.concatenate([
        root_velocity[:, [FORWARD_AXIS, LATERAL_AXIS]],
        np.asarray(root_yaw_rate, dtype=np.float32).reshape(-1, 1),
        positions[:, contact_bones, 1],
    ], axis=1).astype(np.float32)


def styled_local_rotations(rotations: np.ndarray, spec: StyleSpec) -> np.ndarray:
    """
    (n, Bs, 4) parent-local rotations of the styled bones, from character-frame rotations.

    The map :func:`lmm.fk.local_rotations` computes, as one batched product rather than one per
    bone, because this runs on every stage tick; ``test_style_dataset`` holds the two equal.
    """
    rotations = np.asarray(rotations, dtype=np.float64)[:, spec.styled_bones]
    n, n_styled = rotations.shape[0], rotations.shape[1]
    parents = spec.styled_parents

    # The root's parent is the character frame itself, i.e. the identity.
    parent_rotations = np.zeros_like(rotations)
    parent_rotations[..., 3] = 1.0
    has_parent = parents >= 0
    parent_rotations[:, has_parent] = rotations[:, parents[has_parent]]

    local = (Rotation.from_quat(parent_rotations.reshape(-1, 4)).inv()
             * Rotation.from_quat(rotations.reshape(-1, 4))).as_quat()
    return local.reshape(n, n_styled, 4).astype(np.float32)


def input_vectors(local_6d: np.ndarray, root_height: np.ndarray, descriptors: np.ndarray,
                  history: np.ndarray, phase: np.ndarray, phase_rate: np.ndarray) -> np.ndarray:
    """
    ``[styled local 6D | bone-0 height | descriptor | sin, cos phase, phase rate | history]``.

    :param local_6d: (n, Bs, 6). :param root_height: (n,). :param descriptors: (n, D).
    :param history: (n, H, D), in history-offset order. :param phase: (n,) causal phase.
    :param phase_rate: (n,) its rate, rad/s.
    """
    n = local_6d.shape[0]
    phase = np.asarray(phase, dtype=np.float64).reshape(n)
    return np.concatenate([
        np.asarray(local_6d, dtype=np.float32).reshape(n, -1),
        np.asarray(root_height, dtype=np.float32).reshape(n, 1),
        np.asarray(descriptors, dtype=np.float32).reshape(n, -1),
        np.stack([np.sin(phase), np.cos(phase),
                  np.asarray(phase_rate, dtype=np.float64).reshape(n)], axis=1),
        np.asarray(history, dtype=np.float32).reshape(n, -1),
    ], axis=1).astype(np.float32)


def gait_features(descriptors: np.ndarray, history: np.ndarray) -> np.ndarray:
    """
    (n, 2) ``(forward, |lateral|)`` velocity, averaged over the current and history descriptors.

    Lateral is unsigned because a side-walk clip steps both ways, so a signed centroid would
    average to standing. Averaging over the history window keeps the weights from jittering
    within a stride.
    """
    window = np.concatenate([np.asarray(descriptors)[:, np.newaxis, :2],
                             np.asarray(history)[:, :, :2]], axis=1).astype(np.float64)
    return np.stack([window[..., 0].mean(axis=1), np.abs(window[..., 1]).mean(axis=1)], axis=1)


def gait_weights(features: np.ndarray, centroids: np.ndarray, sigma: float) -> np.ndarray:
    """
    (n, G) soft gait weights, a Gaussian kernel on :func:`gait_features`, summing to 1.

    Normalised in log space, so an input far from every centroid still gets weights rather than
    a division by zero.
    """
    features = np.asarray(features, dtype=np.float64).reshape(-1, 2)
    squared = ((features[:, np.newaxis, :] - np.asarray(centroids)[np.newaxis]) ** 2).sum(-1)
    logits = -squared / (2.0 * float(sigma) ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def blend_posture(table: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Soft-weighted posture rotations.

    :param table: (m, G, Bs, 4) one style's per-gait posture per sample.
    :param weights: (m, G) gait weights.
    :return: (m, Bs, 4) xyzw.
    """
    table = np.asarray(table, dtype=np.float64)
    per_bone = np.broadcast_to(np.asarray(weights, dtype=np.float64)[:, :, np.newaxis],
                               table.shape[:-1])
    return blend_quaternions(table, per_bone, axis=1)


def gate_vectors(descriptors: np.ndarray, phase_rate: np.ndarray) -> np.ndarray:
    """``[descriptor | causal phase rate]`` per frame, what the gate's novelty is measured on."""
    phase_rate = np.asarray(phase_rate, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([np.asarray(descriptors, dtype=np.float32)
                           .reshape(phase_rate.shape[0], -1), phase_rate], axis=1)


def history_indices(clip_ranges: np.ndarray, n_frames: int, history_frames: np.ndarray) \
        -> np.ndarray:
    """(n, H) the frame each history offset reads, clamped to its clip's first frame."""
    starts = np.zeros(n_frames, dtype=np.int64)
    for start, end in clip_ranges:
        starts[start:end] = start
    frames = np.arange(n_frames, dtype=np.int64)[:, np.newaxis] - history_frames[np.newaxis, :]
    return np.maximum(frames, starts[:, np.newaxis])


def planar_speed(root_velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(root_velocity, dtype=np.float64)
    return np.hypot(velocity[:, FORWARD_AXIS], velocity[:, LATERAL_AXIS])


def causal_phases(training_set: TrainingSet) -> tuple[np.ndarray, np.ndarray]:
    """The causal phase and rate of every frame, run clip by clip from each clip's first frame."""
    phase = np.zeros(training_set.n_frames, dtype=np.float32)
    rate = np.zeros(training_set.n_frames, dtype=np.float32)
    speed = planar_speed(training_set.root_velocity)
    for start, end in training_set.clip_ranges:
        phase[start:end], rate[start:end] = causal_phase(
            training_set.contacts[start:end] > 0.5, speed[start:end], training_set.frame_time)
    return phase, rate


@dataclass
class FrameInputs:
    """Every frame's network input and what is derived from it, indexed by global frame."""

    descriptors: np.ndarray    # (n, D)
    history: np.ndarray        # (n, H, D)
    phase: np.ndarray          # (n,) causal
    phase_rate: np.ndarray     # (n,) causal
    local_quaternions: np.ndarray  # (n, Bs, 4) styled-bone parent-local rotations
    local_6d: np.ndarray       # (n, Bs, 6)
    root_height: np.ndarray    # (n,)
    x: np.ndarray              # (n, input_size)
    gait_features: np.ndarray  # (n, 2)


def frame_inputs(training_set: TrainingSet, spec: StyleSpec) -> FrameInputs:
    descriptors = descriptor(training_set.positions, training_set.root_velocity,
                             training_set.root_yaw_rate, spec.contact_bones)
    history = descriptors[history_indices(training_set.clip_ranges, training_set.n_frames,
                                          spec.history_frames)]
    phase, phase_rate = causal_phases(training_set)
    local = styled_local_rotations(training_set.rotations, spec)
    local_6d = rotations_to_6d(local)
    root_height = training_set.positions[:, 0, 1].astype(np.float32)
    x = input_vectors(local_6d, root_height, descriptors, history, phase, phase_rate)
    return FrameInputs(descriptors, history, phase, phase_rate, local, local_6d, root_height, x,
                       gait_features(descriptors, history))


# Style filter ----------------------------------------------------------------------------------

@dataclass
class CadenceMeasure:
    """How one (style, gait) steps, over all its clips."""

    footfall_rate: float      # footfalls per second
    duty_factor: float        # mean fraction of time a contact bone is planted
    repeated_fraction: float  # consecutive footfalls on the same foot, as a fraction
    footfalls: int


def _footfalls(contacts: np.ndarray, frame_time: float, min_interval: float) -> list:
    """``(frame, foot)`` rising edges, debounced per foot, in time order."""
    min_frames = min_interval / frame_time
    last = [-math.inf, -math.inf]
    events = []
    for frame in range(1, contacts.shape[0]):
        for foot in (0, 1):
            if contacts[frame, foot] and not contacts[frame - 1, foot] \
                    and frame - last[foot] >= min_frames:
                events.append((frame, foot))
                last[foot] = frame
    return events


def measure_cadence(training_set: TrainingSet, labels: list[ClipLabel],
                    min_interval: float = 0.15) -> dict:
    """``{(style_key, gait): CadenceMeasure}`` over every non-transition clip."""
    totals = {}
    contacts = training_set.contacts > 0.5
    for (start, end), label in zip(training_set.clip_ranges, labels):
        if label.is_transition:
            continue
        clip = contacts[start:end]
        events = _footfalls(clip, training_set.frame_time, min_interval)
        repeats = sum(1 for a, b in zip(events, events[1:]) if a[1] == b[1])
        entry = totals.setdefault((style_key(label.style), label.gait), [0, 0.0, 0.0, 0, 0])
        entry[0] += len(events)
        entry[1] += (end - start) * training_set.frame_time
        entry[2] += float(clip.mean()) * (end - start)
        entry[3] += repeats
        entry[4] += max(0, len(events) - 1)

    measures = {}
    for key, (footfalls, seconds, planted, repeats, transitions) in totals.items():
        frames = seconds / training_set.frame_time
        measures[key] = CadenceMeasure(
            footfall_rate=footfalls / seconds if seconds > 0 else 0.0,
            duty_factor=planted / frames if frames > 0 else 0.0,
            repeated_fraction=repeats / transitions if transitions > 0 else 0.0,
            footfalls=footfalls)
    return measures


def cadence_failures(styled: CadenceMeasure, neutral: CadenceMeasure) -> list[str]:
    """Why a styled gait steps unlike neutral beyond tolerance; empty when it does not."""
    reasons = []
    ratio = styled.footfall_rate / neutral.footfall_rate
    if not CADENCE_RATIO_RANGE[0] <= ratio <= CADENCE_RATIO_RANGE[1]:
        reasons.append(f'footfall rate {ratio:.2f}x neutral')
    if abs(styled.duty_factor - neutral.duty_factor) > DUTY_FACTOR_TOLERANCE:
        reasons.append(f'duty factor {styled.duty_factor:.2f} (neutral {neutral.duty_factor:.2f})')
    if styled.repeated_fraction - neutral.repeated_fraction > REPEATED_FOOT_TOLERANCE:
        reasons.append(f'repeated foot {styled.repeated_fraction:.2f} '
                       f'(neutral {neutral.repeated_fraction:.2f})')
    return reasons


def filter_styles(training_set: TrainingSet, labels: list[ClipLabel], neutral_style: str,
                  excluded_styles=()) -> tuple[list[str], dict]:
    """
    Which styles to train on.

    A style is dropped when the config excludes it, when none of its gaits has a neutral
    counterpart, or when at least half of its measurable gaits step unlike neutral -- a
    cadence-defined style such as Hop, which a stage that keeps the method's timing cannot render.

    :return: ``(kept, dropped)``: kept style names in first-appearance order, neutral included
        and first; dropped as ``{name: reason}``.
    :raises ValueError: the neutral style has no clips, or is itself excluded.
    """
    neutral = style_key(neutral_style)
    excluded = {style_key(name) for name in excluded_styles}
    if neutral in excluded:
        raise ValueError(f'the neutral style {neutral_style!r} cannot be excluded')

    names = {}
    gaits = {}
    for label in labels:
        if label.is_transition:
            continue
        names.setdefault(style_key(label.style), label.style)
        gaits.setdefault(style_key(label.style), set()).add(label.gait)
    if neutral not in names:
        raise ValueError(f'no clips of the neutral style {neutral_style!r} in this database')

    measures = measure_cadence(training_set, labels)
    neutral_gaits = gaits[neutral]

    kept, dropped = [names[neutral]], {}
    for key, name in names.items():
        if key == neutral:
            continue
        if key in excluded:
            dropped[name] = 'excluded by the config'
            continue

        shared = sorted(gaits[key] & neutral_gaits, key=GAITS.index)
        if not shared:
            dropped[name] = 'no gait in common with the neutral style'
            continue

        measured, failures = 0, []
        for gait in shared:
            reference = measures.get((neutral, gait))
            if gait == IDLE_GAIT or reference is None \
                    or reference.footfall_rate < MIN_MEASURABLE_FOOTFALL_RATE:
                continue
            measured += 1
            reasons = cadence_failures(measures[(key, gait)], reference)
            if reasons:
                failures.append(f'{gait}: {", ".join(reasons)}')

        if measured and 2 * len(failures) >= measured:
            dropped[name] = 'cadence-defined: ' + '; '.join(failures)
            continue
        kept.append(name)

    return kept, dropped


# Pairing ---------------------------------------------------------------------------------------

@dataclass
class Pairs:
    """
    Training pairs. ``input`` is the neutral frame the network sees, ``target`` the styled frame.

    ``next`` is the pair of ``target + 1`` in the same clip and split, or -1, for the temporal
    losses. ``validation`` marks pairs whose target lies in its clip's held-out tail.
    """

    input: np.ndarray
    target: np.ndarray
    style: np.ndarray
    gait: np.ndarray
    next: np.ndarray
    validation: np.ndarray

    @property
    def size(self) -> int:
        return int(self.input.size)


def _frames_by_clip_label(training_set: TrainingSet, labels, predicate) -> np.ndarray:
    frames = [np.arange(start, end) for (start, end), label
              in zip(training_set.clip_ranges, labels) if predicate(label)]
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int64)


def pair_frames(training_set: TrainingSet, labels: list[ClipLabel], styles: list[str],
                gait_names: list[str], validation_fraction: float = 0.1) -> Pairs:
    """
    Phase-match every styled frame to a neutral frame of the same gait; see the module docstring.

    Frames whose baked phase rate is zero -- no measurable gait cycle -- are neither paired nor
    used as candidates. ``styles[0]`` is the neutral style, paired with itself.
    """
    style_index = {style_key(name): i for i, name in enumerate(styles)}
    neutral = style_key(styles[0])
    gait_index = {name: i for i, name in enumerate(gait_names)}

    usable = np.asarray(training_set.phase_rate) != 0.0
    phase = np.asarray(training_set.phase, dtype=np.float64)
    circle = np.stack([np.cos(phase), np.sin(phase)], axis=1)
    planar = np.asarray(training_set.root_velocity, dtype=np.float64)[:, [FORWARD_AXIS,
                                                                          LATERAL_AXIS]]

    validation = np.zeros(training_set.n_frames, dtype=bool)
    for start, end in training_set.clip_ranges:
        held = int(round((end - start) * validation_fraction))
        validation[end - held:end] = True

    inputs, targets, style_ids, gait_ids = [], [], [], []
    for gait in gait_names:
        candidates = _frames_by_clip_label(
            training_set, labels, lambda label: style_key(label.style) == neutral
            and label.gait == gait)
        candidates = candidates[usable[candidates]]
        if candidates.size == 0:
            continue
        tree = cKDTree(circle[candidates])
        k = min(PAIRING_CANDIDATES, candidates.size)

        for key, s in style_index.items():
            styled = _frames_by_clip_label(
                training_set, labels, lambda label: style_key(label.style) == key
                and label.gait == gait)
            styled = styled[usable[styled]]
            if styled.size == 0:
                continue

            if key == neutral:
                matched = styled
            else:
                distance, nearest = tree.query(circle[styled], k=k)
                distance = distance.reshape(styled.size, k)
                nearest = nearest.reshape(styled.size, k)
                tie = distance <= distance[:, :1] + PHASE_TIE_TOLERANCE
                velocity_gap = np.linalg.norm(planar[candidates[nearest]]
                                              - planar[styled][:, np.newaxis], axis=2)
                choice = np.where(tie, velocity_gap, np.inf).argmin(axis=1)
                matched = candidates[nearest[np.arange(styled.size), choice]]

            inputs.append(matched)
            targets.append(styled)
            style_ids.append(np.full(styled.size, s, dtype=np.int64))
            gait_ids.append(np.full(styled.size, gait_index[gait], dtype=np.int64))

    if not inputs:
        raise ValueError('no frame could be paired: no styled frame shares a gait with a usable '
                         'neutral frame')

    pairs_input = np.concatenate(inputs).astype(np.int64)
    pairs_target = np.concatenate(targets).astype(np.int64)
    order = np.argsort(pairs_target, kind='stable')
    pairs_input, pairs_target = pairs_input[order], pairs_target[order]
    pairs_style = np.concatenate(style_ids)[order]
    pairs_gait = np.concatenate(gait_ids)[order]

    clip_of = np.full(training_set.n_frames, -1, dtype=np.int64)
    for c, (start, end) in enumerate(training_set.clip_ranges):
        clip_of[start:end] = c
    pair_of_target = np.full(training_set.n_frames, -1, dtype=np.int64)
    pair_of_target[pairs_target] = np.arange(pairs_target.size)

    successor = np.minimum(pairs_target + 1, training_set.n_frames - 1)
    next_pair = pair_of_target[successor]
    same = (successor == pairs_target + 1) & (clip_of[successor] == clip_of[pairs_target]) \
        & (validation[successor] == validation[pairs_target])
    next_pair = np.where(same, next_pair, -1)

    return Pairs(pairs_input, pairs_target, pairs_style, pairs_gait, next_pair,
                 validation[pairs_target])


# Closed-form terms -----------------------------------------------------------------------------

@dataclass
class ClosedForm:
    """
    The per-(style, gait) tables fitted without a network.

    :param gait_names: the gaits the neutral style has, in :data:`GAITS` order.
    :param gait_centroids: (G, 2) neutral :func:`gait_features` mean per gait.
    :param posture_rotations: (S, G, Bs, 4) mean parent-local delta, xyzw.
    :param posture_heights: (S, G) mean bone-0 height offset, metres.
    :param speed_ratios: (S, G) styled over neutral mean planar speed.
    """

    gait_names: list
    gait_centroids: np.ndarray
    gait_sigma: float
    posture_rotations: np.ndarray
    posture_heights: np.ndarray
    speed_ratios: np.ndarray
    fitted: np.ndarray = field(default=None)  # (S, G) bool: estimated rather than filled in


def neutral_gaits(labels: list[ClipLabel], neutral_style: str) -> list[str]:
    """The gaits the neutral style has clips for, in :data:`GAITS` order."""
    present = {label.gait for label in labels
               if style_key(label.style) == style_key(neutral_style) and not label.is_transition}
    return [gait for gait in GAITS if gait in present]


def gait_centroids(training_set: TrainingSet, labels: list[ClipLabel], neutral_style: str,
                   gait_names: list[str], features: np.ndarray) -> np.ndarray:
    centroids = np.zeros((len(gait_names), 2), dtype=np.float64)
    for g, gait in enumerate(gait_names):
        frames = _frames_by_clip_label(
            training_set, labels, lambda label: style_key(label.style) == style_key(neutral_style)
            and label.gait == gait)
        centroids[g] = features[frames].mean(axis=0)
    return centroids.astype(np.float32)


def fit_closed_form(training_set: TrainingSet, pairs: Pairs, inputs: FrameInputs,
                    n_styles: int, gait_names: list[str], centroids: np.ndarray,
                    gait_sigma: float) -> ClosedForm:
    """
    Posture means and speed ratios from the **training** pairs.

    ``Neutral`` (style 0) is the identity by definition. A (style, gait) with no pairs takes its
    style's mean over the gaits it does have, so a soft weight landing there is still sensible.
    """
    n_gaits, n_styled = len(gait_names), inputs.local_quaternions.shape[1]
    rotations = np.zeros((n_styles, n_gaits, n_styled, 4), dtype=np.float64)
    rotations[..., 3] = 1.0
    heights = np.zeros((n_styles, n_gaits), dtype=np.float64)
    ratios = np.ones((n_styles, n_gaits), dtype=np.float64)
    fitted = np.zeros((n_styles, n_gaits), dtype=bool)
    fitted[0] = True

    train = ~pairs.validation
    speed = planar_speed(training_set.root_velocity)

    # Over every neutral frame of the gait rather than the matched ones: the velocity tie-break
    # picks neutral frames close to the styled speed, which would pull every ratio toward 1.
    neutral_speeds = np.zeros(n_gaits)
    for g in range(n_gaits):
        chosen = train & (pairs.style == 0) & (pairs.gait == g)
        neutral_speeds[g] = float(np.mean(speed[pairs.target[chosen]])) if chosen.any() else 0.0

    for s in range(1, n_styles):
        counts = np.zeros(n_gaits)
        for g in range(n_gaits):
            chosen = train & (pairs.style == s) & (pairs.gait == g)
            if not chosen.any():
                continue
            source = inputs.local_quaternions[pairs.input[chosen]].reshape(-1, 4)
            target = inputs.local_quaternions[pairs.target[chosen]].reshape(-1, 4)
            delta = (Rotation.from_quat(source).inv() * Rotation.from_quat(target)).as_quat()
            rotations[s, g] = blend_quaternions(delta.reshape(-1, n_styled, 4),
                                                np.ones(int(chosen.sum())), axis=0)
            heights[s, g] = float(np.mean(inputs.root_height[pairs.target[chosen]]
                                          - inputs.root_height[pairs.input[chosen]]))
            counts[g] = chosen.sum()
            fitted[s, g] = True

            if gait_names[g] != IDLE_GAIT and neutral_speeds[g] >= MIN_RATIO_SPEED:
                ratios[s, g] = np.clip(float(np.mean(speed[pairs.target[chosen]]))
                                       / neutral_speeds[g], *SPEED_RATIO_RANGE)

        if counts.sum() == 0:
            continue
        present = counts > 0
        style_mean = blend_quaternions(rotations[s, present], counts[present], axis=0)
        mean_height = float(np.average(heights[s, present], weights=counts[present]))
        for g in np.flatnonzero(~present):
            rotations[s, g] = style_mean
            heights[s, g] = mean_height

    return ClosedForm(gait_names=list(gait_names), gait_centroids=np.asarray(centroids),
                      gait_sigma=float(gait_sigma),
                      posture_rotations=rotations.astype(np.float32),
                      posture_heights=heights.astype(np.float32),
                      speed_ratios=ratios.astype(np.float32), fitted=fitted)


@dataclass
class PairTargets:
    """Per-pair soft-weighted closed-form terms and the root residual target."""

    posture_6d: np.ndarray     # (m, Bs, 6) soft-weighted posture rotation
    posture_height: np.ndarray  # (m,)
    speed_ratio: np.ndarray    # (m,)
    root_residual: np.ndarray  # (m, 4): v_styled - k * v_neutral (3), yaw rate difference


def pair_targets(training_set: TrainingSet, pairs: Pairs, inputs: FrameInputs,
                 closed_form: ClosedForm, chunk: int = 16384) -> PairTargets:
    """
    The soft posture, height and speed ratio each pair's input implies, and the root residual
    target ``(v_styled - k * v_neutral, yaw_rate_styled - yaw_rate_neutral)``.

    The posture is the same blend :func:`blend_posture` gives the runtime, so the dynamics target
    the trainer derives from it is the remainder the runtime composes with.
    """
    weights = gait_weights(inputs.gait_features[pairs.input], closed_form.gait_centroids,
                           closed_form.gait_sigma)
    n_styled = closed_form.posture_rotations.shape[2]
    posture = np.zeros((pairs.size, n_styled, 4), dtype=np.float32)
    for start in range(0, pairs.size, chunk):
        stop = min(start + chunk, pairs.size)
        posture[start:stop] = blend_posture(
            closed_form.posture_rotations[pairs.style[start:stop]], weights[start:stop])

    heights = (closed_form.posture_heights[pairs.style] * weights).sum(axis=1)
    ratios = (closed_form.speed_ratios[pairs.style] * weights).sum(axis=1)

    velocity = np.asarray(training_set.root_velocity, dtype=np.float64)
    yaw = np.asarray(training_set.root_yaw_rate, dtype=np.float64)
    residual = np.concatenate([
        velocity[pairs.target] - ratios[:, np.newaxis] * velocity[pairs.input],
        (yaw[pairs.target] - yaw[pairs.input])[:, np.newaxis],
    ], axis=1)

    return PairTargets(rotations_to_6d(posture), heights.astype(np.float32),
                       ratios.astype(np.float32), residual.astype(np.float32))
