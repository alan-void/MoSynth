"""
Packs a :class:`training_data.TrainingSet` into the input and output vectors a PFNN trains on.

:mod:`training_data` deliberately stops one step short of this, because the packing is a property
of the model rather than of the database. This module is that step for a phase-functioned network
(Holden et al. 2017): given a query frame, the network sees a dense trajectory window either side
of it, the joints as they stand, and a gait phase; it predicts the pose one frame later, the root
delta that carries the character there, and how far the phase advanced.

Two deliberate departures from the paper, both forced by what this repository already is:

* **Rotations, not positions.** The paper predicts joint positions and reconciles them with IK.
  ``PoseBuffer`` is rotation-based and there is no IK solver here, so the output carries 6D
  rotations (Zhou et al.) and the positions the next input needs come back from forward kinematics.
  That also means a predicted pose cannot violate a bone length, which a position-space prediction
  can and does.
* **Foot contacts in place of a gait label.** The paper feeds a one-hot gait vector authored per
  clip. Nothing here is labelled that way, but the contacts the phase was reconstructed from are
  stored, and they carry the part of that signal a locomotion model can use.

Because rotations are what is predicted, a bone is only meaningful to predict when its parent is
predicted too -- otherwise there is no frame to apply the rotation in. :func:`select_bones` enforces
that, and the Unity side keeps it true by excluding a bone's whole subtree along with it.

Conventions are inherited unchanged from :mod:`training_data`: quaternions xyzw, y-up left-handed
with the character facing +z, every rate per second, and the ground plane written as ``(x, z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training_data import TrainingSet, rotations_to_6d

# Frames either side of the query frame the default window spans -- one second at 30 fps, which is
# the horizon the paper uses. Wide enough that a turn is visible before the character reaches it.
DEFAULT_WINDOW_RADIUS = 30

# Frames between adjacent samples of the default window, giving 13 samples across it. Denser costs
# input width for very little new information, since the trajectory is smooth at this scale.
DEFAULT_WINDOW_STRIDE = 5


def default_window_offsets(radius: int = DEFAULT_WINDOW_RADIUS,
                           stride: int = DEFAULT_WINDOW_STRIDE) -> np.ndarray:
    """Symmetric frame offsets from ``-radius`` to ``+radius``, always including 0."""
    if radius < 0 or stride < 1:
        raise ValueError(f'radius must be >= 0 and stride >= 1, got {radius} and {stride}')
    half = np.arange(stride, radius + 1, stride, dtype=np.int64)
    return np.concatenate([-half[::-1], [0], half])


def select_bones(bone_names, parents: np.ndarray, excluded_names) -> np.ndarray:
    """
    The bones a model predicts: every bone the caller did not exclude, in skeleton order.

    The exclusion list is authored on the Unity config and travels here by name, so a joint that
    moves in the hierarchy keeps its setting. Names it does not recognise are an error rather than
    a silent no-op: a misspelt exclusion would quietly train a different model than intended, and
    the symptom would not appear until the checkpoint failed to load.

    :param bone_names: every bone of the database, in depth-first order.
    :param parents: (n_bones,) parent index per bone, -1 for the root.
    :param excluded_names: bone names not to predict.
    :raises ValueError: if a name is unknown, if the root is excluded, or if the remaining set is
        not closed under parent -- see this module's docstring for why rotations require that.
    """
    bone_names = list(bone_names)
    excluded = set(excluded_names)

    unknown = sorted(excluded.difference(bone_names))
    if unknown:
        raise ValueError(f'excluded bones not in this skeleton: {", ".join(unknown)}')

    keep = np.array([name not in excluded for name in bone_names], dtype=bool)
    if not keep[0]:
        raise ValueError(f'the root bone {bone_names[0]!r} cannot be excluded: it is the frame '
                         'every other rotation is resolved against')

    orphans = [bone_names[i] for i in range(1, len(bone_names))
               if keep[i] and not keep[parents[i]]]
    if orphans:
        raise ValueError(
            'these bones are kept but their parent is excluded, so there is no frame to apply '
            f'their rotation in: {", ".join(orphans)}. Exclude a bone together with its subtree.')

    return np.flatnonzero(keep).astype(np.int64)


def _layout(blocks) -> list:
    layout, offset = [], 0
    for name, count in blocks:
        layout.append((name, offset, count))
        offset += count
    return layout


@dataclass(frozen=True)
class PfnnSpec:
    """
    Everything about the packing that a trained model has to be fed back exactly.

    Stored in the checkpoint and checked against the live skeleton at load, for the reason
    ``neural-synthesis.md`` is written around: a training/inference disagreement about which bones
    or which horizons these numbers describe does not throw, it just makes the network wrong.

    :param window_offsets: (T,) frame offsets of the trajectory window, negative for the past.
    :param bone_indices: (B,) database bone indices this model predicts, ascending.
    :param bone_names: the names of those bones, in the same order.
    :param frame_time: seconds per frame of the database it was packed from.
    """

    window_offsets: np.ndarray
    bone_indices: np.ndarray
    bone_names: tuple
    frame_time: float

    @property
    def n_offsets(self) -> int:
        return int(np.size(self.window_offsets))

    @property
    def n_bones(self) -> int:
        return int(np.size(self.bone_indices))

    def input_layout(self) -> list:
        """``(name, offset, float_count)`` per block of an input vector."""
        return _layout([('trajectory_positions', 2 * self.n_offsets),
                        ('trajectory_directions', 2 * self.n_offsets),
                        ('joint_positions', 3 * self.n_bones),
                        ('joint_velocities', 3 * self.n_bones),
                        ('contacts', 2)])

    def output_layout(self) -> list:
        """``(name, offset, float_count)`` per block of an output vector."""
        return _layout([('joint_rotations_6d', 6 * self.n_bones),
                        ('joint_velocities', 3 * self.n_bones),
                        ('root_height', 1),
                        ('root_delta', 3),
                        ('phase_delta', 1),
                        ('contacts', 2)])

    @property
    def input_size(self) -> int:
        _, offset, count = self.input_layout()[-1]
        return offset + count

    @property
    def output_size(self) -> int:
        _, offset, count = self.output_layout()[-1]
        return offset + count


def block(vectors: np.ndarray, layout, name: str) -> np.ndarray:
    """The named block of a packed vector, or of a batch of them, as a view."""
    for entry_name, offset, count in layout:
        if entry_name == name:
            return vectors[..., offset:offset + count]
    raise KeyError(f'no block named {name!r}; have {[n for n, _, _ in layout]}')


def build_spec(training_set: TrainingSet, excluded_bones=(),
               window_radius: int = DEFAULT_WINDOW_RADIUS,
               window_stride: int = DEFAULT_WINDOW_STRIDE) -> PfnnSpec:
    """The packing a model over this database and this bone selection will use."""
    bone_indices = select_bones(training_set.bone_names, training_set.parents, excluded_bones)
    return PfnnSpec(
        window_offsets=default_window_offsets(window_radius, window_stride),
        bone_indices=bone_indices,
        bone_names=tuple(training_set.bone_names[i] for i in bone_indices),
        frame_time=float(training_set.frame_time))


def usable_queries(training_set: TrainingSet) -> np.ndarray:
    """
    (n,) bool: frames that can serve as a query, i.e. that have a successor to predict.

    Three conditions, and the second is the one a caller is likely to forget:
    :meth:`TrainingSet.usable` checks only the matching feature vector, while a phase-functioned
    network is meaningless on a clip with no measurable gait cycle -- which is exactly what
    ``phase_rate == 0`` marks. The third keeps a sample from differencing across a jump cut.
    """
    usable = training_set.usable()
    has_cycle = training_set.phase_rate != 0.0

    has_successor = np.zeros(training_set.n_frames, dtype=bool)
    for start, end in training_set.clip_ranges:
        if end - start >= 2:
            has_successor[start:end - 1] = True

    successor_usable = np.zeros(training_set.n_frames, dtype=bool)
    successor_usable[:-1] = usable[1:]

    return usable & has_cycle & has_successor & successor_usable


def build_vectors(training_set: TrainingSet, spec: PfnnSpec):
    """
    Pack every usable frame into one input and one output vector.

    The query frame ``i`` supplies the input and the frame after it supplies the target, so the
    network learns one database timestep. Everything is already in the character frame of whichever
    frame it belongs to -- ``i`` for the input, ``i + 1`` for the pose half of the output -- and the
    root delta is the transform between those two frames, which is what carries the character.

    :return: ``(x (m, input_size), y (m, output_size), phase (m,), frames (m,))``, all float32
        except ``frames``, which is the database index each sample was taken from.
    """
    queries = np.flatnonzero(usable_queries(training_set)).astype(np.int64)
    nxt = queries + 1
    bones = spec.bone_indices

    positions, directions = training_set.trajectory_window(spec.window_offsets)

    # The step from frame i to frame i+1, expressed in frame i's own space. Read straight out of the
    # sampler rather than re-derived from root_velocity, so there is one definition of "where the
    # character goes next" and it is the one two independent checks were run against.
    step_positions, step_directions = training_set.trajectory_window([1])
    root_delta = np.concatenate([
        step_positions[queries, 0, :],
        np.arctan2(step_directions[queries, 0, 0],
                   step_directions[queries, 0, 1])[:, np.newaxis],
    ], axis=1)

    rotations_6d = rotations_to_6d(training_set.rotations[nxt][:, bones])

    m = queries.size
    x = np.concatenate([
        positions[queries].reshape(m, -1),
        directions[queries].reshape(m, -1),
        training_set.positions[queries][:, bones].reshape(m, -1),
        training_set.velocities[queries][:, bones].reshape(m, -1),
        training_set.contacts[queries],
    ], axis=1).astype(np.float32)

    y = np.concatenate([
        rotations_6d.reshape(m, -1),
        training_set.velocities[nxt][:, bones].reshape(m, -1),
        training_set.positions[nxt][:, 0, 1][:, np.newaxis],
        root_delta,
        (training_set.phase_rate[queries] * training_set.frame_time)[:, np.newaxis],
        training_set.contacts[nxt],
    ], axis=1).astype(np.float32)

    if x.shape[1] != spec.input_size or y.shape[1] != spec.output_size:
        raise AssertionError(
            f'packed {x.shape[1]}/{y.shape[1]} floats but the layout declares '
            f'{spec.input_size}/{spec.output_size}')

    return x, y, training_set.phase[queries].astype(np.float32), queries


def normalization(vectors: np.ndarray):
    """
    ``(mean, std)`` over a set of packed vectors, with constant columns left alone.

    A column that never varies -- a contact flag on a database with no flight phase, say -- would
    otherwise be divided by zero and arrive at the network as an infinity.
    """
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)
