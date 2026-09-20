"""
Packs a :class:`training_data.TrainingSet` into the vectors a Learned Motion Matching model trains
on (Holden et al. 2020).

Four vectors, and the whole method is in how they relate:

* **X** is the matching feature vector, read straight out of the ``.mmfeatures`` the classic matcher
  searches. Not rebuilt here, and deliberately: the point of the method is to replace the *search*
  over that database, so a learned matcher answering a differently-shaped query would not be
  comparable to the matcher it replaces.
* **Y** is the pose as a rig is posed: joint-*local* rotations, the rate channels, the root's height
  and its own motion, and the foot contacts. This is what a decompressor reconstructs.
* **Q** is the same pose after forward kinematics -- where every joint actually ended up, in the
  character frame. Nothing predicts ``Q`` directly; it is derived from ``Y`` on both sides of the
  loss, which is what makes an error deep in a chain cost what it is worth rather than what a
  single joint angle is worth.
* **Z** is a small latent the compressor encodes ``[Y Q]`` into, and the decompressor's second
  input beside ``X``.

``Z`` comes from one frame in *two spaces*, not from two frames. The paper's Algorithm 1 is
``Z <- C([Y Q])``, and its stated reason is that the compressor "was able to copy features directly
to the latent space if it found them useful". Temporal structure in the latent comes from the
velocity regulariser instead, which is what leaves it steppable in phase B.

A frame still only has a latent when its successor is in the same clip -- see :func:`compressible`.
The pair is what the velocity terms of the loss are differenced over, and sampling across a clip
boundary would score a jump cut as motion, which is the rule every window in :mod:`training_data`
follows.

Conventions are inherited unchanged from :mod:`training_data`: quaternions xyzw, y-up left-handed
with the character facing +z, every rate per second, and the ground plane written as ``(x, z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import lmm_fk
import neural_packing
from training_data import TrainingSet, rotations_to_6d

# Floats the latent is compressed to. The paper's section 4.4 and the reference implementation's
# shipped graphs agree on it, over a rig of comparable size.
DEFAULT_LATENT_SIZE = 32


def pose_vector_layout(n_bones: int) -> list:
    """
    ``(name, offset, float_count)`` per block of ``Y``, the decompressor's output.

    Joint-local rotations, both rate channels, and the root's own motion. Two departures from the
    paper's ``Y``, both forced by what this project's runtime consumes:

    * **No joint-local translations.** The rig has fixed bone lengths and
      ``CharacterSpacePose.Apply`` writes the rest offsets regardless, so a predicted translation
      would be discarded at the one place it could be used.
    * **The rate channels stay in the character frame**, which is the convention ``Apply`` reads and
      what :mod:`training_data` already derives. The paper expresses them joint-locally.

    ``root_height`` takes the slot the paper's ``o`` (root offset) occupies: the root's ``x`` and
    ``z`` are what the character frame transform removed, so its height is the only free float.
    """
    return neural_packing.layout([
        ('rotations_6d', 6 * n_bones),
        ('velocities', 3 * n_bones),
        ('angular_velocities', 3 * n_bones),
        ('root_height', 1),
        ('root_velocity', 3),
        ('root_yaw_rate', 1),
        ('contacts', 2),
    ])


def character_vector_layout(n_bones: int) -> list:
    """
    ``(name, offset, float_count)`` per block of ``Q``, the pose after forward kinematics.

    Never stored in a checkpoint and never produced at runtime -- it exists only between the two
    halves of the training loss, and in the compressor's input.
    """
    return neural_packing.layout([
        ('positions', 3 * n_bones),
        ('rotations_6d', 6 * n_bones),
    ])


# What each block of each loss term is worth, in the real units the quantity is measured in.
#
# These are the author's own released training code, not the paper -- the paper states no numbers,
# only "roughly equal weight to all pose-based losses and a small weighting to regularization
# losses". They are applied to *denormalised* differences, so a metre of position error and a radian
# of angular error are genuinely being traded against each other here, which is the only reading
# under which numbers like 75 and 1.25 make sense.
LOCAL_WEIGHTS = {
    'rotations_6d': 10.0,
    'velocities': 10.0,
    'angular_velocities': 1.25,
    'root_height': 75.0,
    'root_velocity': 2.0,
    'root_yaw_rate': 2.0,
    'contacts': 2.0,
}

CHARACTER_WEIGHTS = {
    'positions': 15.0,
    'rotations_6d': 5.0,
}

# The velocity halves of the loss, differenced over the frame pair. Only the rotations are
# differenced locally, because the local rate channels of Y are already rates and scoring their
# own difference would be scoring an acceleration nothing else in the pipeline reads.
LOCAL_RATE_WEIGHTS = {'rotations_6d': 1.75}
CHARACTER_RATE_WEIGHTS = {'positions': 2.0, 'rotations_6d': 0.75}

# The three latent regularisers of Algorithm 1: magnitude, energy, and rate of change. w_vreg looks
# tiny beside the others because it multiplies a *per-second* derivative -- a factor of sixty
# against the per-frame delta it would be easy to write by mistake.
#
# w_vreg is the one number here worth tuning, and the trainer takes it as an argument for that
# reason. It decides whether the latent evolves smoothly enough for the phase B stepper to advance,
# and the two magnitude terms work against it: both shrink |Z|, which shrinks the absolute step
# this penalises, while the *relative* step -- the thing that actually decides steppability -- is
# not penalised by any of the three. So the equilibrium is data-dependent in a way the other
# weights are not.
LATENT_SPARSITY_WEIGHT = 0.1
LATENT_MAGNITUDE_WEIGHT = 0.1
LATENT_VELOCITY_WEIGHT = 0.01


@dataclass(frozen=True)
class LmmSpec:
    """
    Everything about the packing that a trained model has to be fed back exactly.

    Stored in the checkpoint and checked against the live database at load, for the reason the
    neural work here is organised around: a training/inference disagreement about which bones or
    which features these numbers describe does not throw, it just makes the network wrong.

    :param bone_indices: (B,) database bone indices the decompressor predicts, ascending.
    :param bone_names: the names of those bones, in the same order.
    :param parents: (B,) parent index of each, expressed within the selection; -1 for the root.
    :param rest_offsets: (B, 3) each bone's offset from its parent, in the parent's space.
    :param feature_size: floats per matching feature vector, i.e. the width of ``X``.
    :param latent_size: floats per latent, i.e. the width of ``Z``.
    :param frame_time: seconds per frame of the database it was packed from.
    """

    bone_indices: np.ndarray
    bone_names: tuple
    parents: np.ndarray
    rest_offsets: np.ndarray
    feature_size: int
    latent_size: int
    frame_time: float

    @property
    def n_bones(self) -> int:
        return int(np.size(self.bone_indices))

    def pose_layout(self) -> list:
        """``(name, offset, float_count)`` per block of ``Y``."""
        return pose_vector_layout(self.n_bones)

    def character_layout(self) -> list:
        """``(name, offset, float_count)`` per block of ``Q``."""
        return character_vector_layout(self.n_bones)

    @property
    def pose_size(self) -> int:
        """Floats per pose vector -- the decompressor's output width."""
        return neural_packing.vector_size(self.pose_layout())

    @property
    def character_size(self) -> int:
        """Floats per character-space vector."""
        return neural_packing.vector_size(self.character_layout())

    @property
    def compressor_input_size(self) -> int:
        """``Y`` and ``Q`` concatenated: one frame in two spaces."""
        return self.pose_size + self.character_size

    @property
    def decompressor_input_size(self) -> int:
        """``X`` and ``Z`` concatenated, in that order."""
        return self.feature_size + self.latent_size


def build_spec(training_set: TrainingSet, excluded_bones=(),
               latent_size: int = DEFAULT_LATENT_SIZE) -> LmmSpec:
    """
    The packing a model over this database and this bone selection will use.

    :raises ValueError: if the database carries no matching feature vectors. A learned matcher has
        nothing to be a matcher *of* without them -- see :func:`training_data.load_database`.
    """
    if training_set.features is None:
        raise ValueError(
            'this database has no .mmfeatures, so there is no query vector to learn against. '
            'Learned motion matching reads the same features the classic matcher searches; '
            'generate them from a MotionMatchingData asset that authors feature channels.')

    bone_indices = neural_packing.select_bones(
        training_set.bone_names, training_set.parents, excluded_bones)
    parents = lmm_fk.parents_within(training_set.parents, bone_indices)

    return LmmSpec(
        bone_indices=bone_indices,
        bone_names=tuple(training_set.bone_names[i] for i in bone_indices),
        parents=parents,
        rest_offsets=lmm_fk.rest_offsets(training_set.positions[0][bone_indices],
                                         training_set.rotations[0][bone_indices], parents),
        feature_size=int(training_set.features.shape[1]),
        latent_size=int(latent_size),
        frame_time=float(training_set.frame_time))


def pose_vector(training_set: TrainingSet, spec: LmmSpec) -> np.ndarray:
    """Every frame's ``Y``. (n_frames, :attr:`LmmSpec.pose_size`) float32."""
    n = training_set.n_frames
    bones = spec.bone_indices

    return np.concatenate([
        lmm_fk.local_rotations_6d(training_set.rotations[:, bones], spec.parents).reshape(n, -1),
        training_set.velocities[:, bones].reshape(n, -1),
        training_set.angular_velocities[:, bones].reshape(n, -1),
        training_set.positions[:, bones[0], 1].reshape(n, 1),
        training_set.root_velocity,
        training_set.root_yaw_rate.reshape(n, 1),
        training_set.contacts,
    ], axis=1).astype(np.float32)


def character_vector(training_set: TrainingSet, spec: LmmSpec) -> np.ndarray:
    """
    Every frame's ``Q``. (n_frames, :attr:`LmmSpec.character_size`) float32.

    Read out of the training set rather than run through :mod:`lmm_fk`, because the training set is
    already in the character frame -- the FK there exists to derive ``Q`` from a *prediction*, and
    ``test_lmm_fk`` is what holds the two definitions to each other.
    """
    n = training_set.n_frames
    bones = spec.bone_indices

    return np.concatenate([
        training_set.positions[:, bones].reshape(n, -1),
        rotations_to_6d(training_set.rotations[:, bones]).reshape(n, -1),
    ], axis=1).astype(np.float32)


def compressible(training_set: TrainingSet) -> np.ndarray:
    """
    (n,) bool: frames that have a latent, i.e. that have a successor in the same clip.

    Usability is :meth:`TrainingSet.usable`, which is feature validity and nothing else.
    :func:`pfnn_dataset.usable_queries` looks interchangeable with this and is not -- its
    ``phase_rate != 0`` filter discards every clip with no measurable gait cycle, which a
    phase-functioned network needs and a learned matcher wants kept. Idling is motion a matcher
    has to be able to produce.
    """
    usable = training_set.usable()

    has_successor = np.zeros(training_set.n_frames, dtype=bool)
    for start, end in training_set.clip_ranges:
        if end - start >= 2:
            has_successor[start:end - 1] = True

    successor_usable = np.zeros(training_set.n_frames, dtype=bool)
    successor_usable[:-1] = usable[1:]

    return usable & has_successor & successor_usable


def training_pairs(training_set: TrainingSet) -> np.ndarray:
    """
    (m,) frame indices ``i`` where both ``i`` and ``i + 1`` have a latent.

    The autoencoder trains on pairs rather than on single frames because three terms of the loss --
    the two velocity terms and the latent's own -- are differences over adjacent frames. Each pair
    contributes both of its frames to the reconstruction terms, so the only frames lost are the last
    two of every clip.
    """
    latent_exists = compressible(training_set)

    next_exists = np.zeros(training_set.n_frames, dtype=bool)
    next_exists[:-1] = latent_exists[1:]

    return np.flatnonzero(latent_exists & next_exists).astype(np.int64)


def build_vectors(training_set: TrainingSet, spec: LmmSpec):
    """
    Pack the whole database once, as the arrays training indexes into.

    Everything is returned per database frame rather than per sample, because the autoencoder,
    the latent bake and the phase B stepper all index the same frames differently and copying a
    gathered subset for each would cost more than the gather.

    :return: ``(x (n, feature_size), y (n, pose_size), q (n, character_size),
        latent_exists (n,) bool)``. Rows of ``x`` where the feature vector is invalid are zero, as
        Unity wrote them, and are excluded by ``latent_exists``.
    """
    x = np.ascontiguousarray(training_set.features, dtype=np.float32)
    y = pose_vector(training_set, spec)
    q = character_vector(training_set, spec)

    if x.shape[1] != spec.feature_size:
        raise AssertionError(f'database features are {x.shape[1]} floats but the spec declares '
                             f'{spec.feature_size}')
    if y.shape[1] != spec.pose_size:
        raise AssertionError(f'packed {y.shape[1]} pose floats but the layout declares '
                             f'{spec.pose_size}')
    if q.shape[1] != spec.character_size:
        raise AssertionError(f'packed {q.shape[1]} character floats but the layout declares '
                             f'{spec.character_size}')

    return x, y, q, compressible(training_set)


def group_normalization(vectors: np.ndarray, block_layout):
    """
    ``(mean, std)`` over packed vectors: a mean per float, but **one standard deviation per block**.

    The per-block scalar is the paper's scheme -- it scales "each feature (e.g. left foot position)"
    by a single number -- and the difference from a per-float std is not cosmetic. Dividing a
    3-vector by three different numbers warps it: the axes are no longer comparable, so an L1 error
    on the result means something different per axis and the near-isotropy of a joint's motion is
    destroyed on the way in. A block whose floats never vary is left alone rather than divided by
    zero.

    :param vectors: (n, vector_size) the packed vectors to measure.
    :param block_layout: ``(name, offset, count)`` per block.
    """
    mean = vectors.mean(axis=0)
    std = np.ones(vectors.shape[1], dtype=np.float32)

    for _, offset, count in block_layout:
        if count == 0:
            continue
        spread = float(vectors[:, offset:offset + count].std())
        std[offset:offset + count] = spread if spread > 1e-6 else 1.0

    return mean.astype(np.float32), std


def feature_weights(training_set: TrainingSet, authored) -> np.ndarray:
    """
    Expand one authored weight per feature definition into one weight per float of ``X``.

    The same expansion ``MotionMatchingStage.UpdateFeatureWeights`` does, and it has to stay the
    same one: a learned matcher fitted against uniform weights approximates a search nobody runs,
    and the comparison against the classic matcher is then silently measuring two different things.

    :param authored: one weight per feature definition, trajectory features first, in the order the
        ``.mmfeatures`` schema lists them. A short list is padded with ones.
    :raises ValueError: if the database carries no feature schema to expand against.
    """
    if training_set.feature_widths is None or training_set.feature_counts is None:
        raise ValueError('this database carries no feature schema, so per-float weights cannot be '
                         'derived. Regenerate it from a MotionMatchingData asset.')

    widths = np.asarray(training_set.feature_widths, dtype=np.int64)
    counts = np.asarray(training_set.feature_counts, dtype=np.int64)
    authored = np.asarray(list(authored), dtype=np.float32)

    if authored.size < widths.size:
        authored = np.concatenate([authored, np.ones(widths.size - authored.size, dtype=np.float32)])

    return np.repeat(authored[:widths.size], widths * counts).astype(np.float32)
