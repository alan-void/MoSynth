"""
Packs a :class:`training.training_data.TrainingSet` into the vectors a Learned Motion Matching model trains
on (Holden et al. 2020).

Four vectors:

* **X** is the matching feature vector, read straight out of the ``.mmfeatures`` the classic matcher
  searches, so the learned matcher answers the same query as the search it replaces.
* **Y** is the pose as a rig is posed: joint-*local* rotations, the rate channels, the root's height
  and its own motion, and the foot contacts. This is what a decompressor reconstructs.
* **Q** is the same pose after forward kinematics, in the character frame. Nothing predicts ``Q``
  directly; it is derived from ``Y`` on both sides of the loss, so an error deep in a chain costs
  what it is worth.
* **Z** is a small latent, the decompressor's second input beside ``X``. The compressor encodes
  it from ``[Y Q]`` -- one frame in two spaces, not two frames, per the paper's Algorithm 1.

A frame only has a latent when its successor is in the same clip -- see :func:`compressible`.
See openwiki/motion-matching/learned-motion-matching.md for the method in full.

Conventions are inherited unchanged from :mod:`training.training_data`: quaternions xyzw, y-up
left-handed with the character facing +z, every rate per second, and the ground plane written as
``(x, z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lmm import fk
from training import neural_packing
from training.training_data import TrainingSet, rotations_to_6d

# Floats the latent is compressed to. The paper's section 4.4 and the reference implementation's
# shipped graphs agree on it, over a rig of comparable size.
DEFAULT_LATENT_SIZE = 32


def pose_vector_layout(n_bones: int) -> list:
    """
    ``(name, offset, float_count)`` per block of ``Y``, the decompressor's output.

    Two departures from the paper's ``Y``, both forced by what ``CharacterSpacePose.Apply``
    consumes: no joint-local translations (it writes rest offsets regardless), and the rate channels
    stay in the character frame. ``root_height`` takes the paper's root-offset slot, since the
    root's ``x`` and ``z`` are what the character frame removed.
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


# What each block of each loss term is worth, taken from the paper author's released training
# code. Applied to *denormalised* differences, so metres and radians are traded directly.
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
# differenced locally: Y's rate channels are already rates, and their difference is an acceleration
# nothing reads.
LOCAL_RATE_WEIGHTS = {'rotations_6d': 1.75}
CHARACTER_RATE_WEIGHTS = {'positions': 2.0, 'rotations_6d': 0.75}

# The three latent regularisers of Algorithm 1: magnitude, energy, and rate of change. The rate
# weight (w_vreg) multiplies a *per-second* derivative, hence its small size. It is the one to tune,
# so the trainer takes it as an argument; see openwiki/motion-matching/learned-motion-matching.md.
LATENT_SPARSITY_WEIGHT = 0.1
LATENT_MAGNITUDE_WEIGHT = 0.1
LATENT_VELOCITY_WEIGHT = 0.01

# What each half of the stepper's answer is worth, from the released training code. The latent
# outweighs the features because only the next accepted candidate corrects a drifted latent, while
# the controller re-supplies features at every search. Rate terms are per-second and arrive
# large, hence their small weights.
STEPPER_WEIGHTS = {
    'features': 2.0,
    'latent': 7.5,
    'feature_rate': 0.2,
    'latent_rate': 0.5,
}

# Frames the stepper is unrolled over while training: two searches' worth at the stage's default
# ``searchInterval`` of ten frames, and the window the released code uses.
DEFAULT_STEPPER_WINDOW = 20

# What each term of the projector's answer is worth, from the released training code. A wrong Z is
# a pose the decompressor has never been asked for, so the latent dominates; the distance term is a
# scalar beside two whole vectors, hence small.
PROJECTOR_WEIGHTS = {
    'features': 1.0,
    'latent': 5.0,
    'distance': 0.3,
}

# Added to each feature's own spread to get the scale the projector's training noise is drawn at,
# so a near-constant feature is still displaced. X is already normalised, so this roughly doubles a
# unit scale.
PROJECTOR_NOISE_FLOOR = 1.0

# Per-sample noise is drawn at ``sigma ~ U(0, PROJECTOR_SIGMA)`` times that scale, so one batch
# spans a query that is nearly a database frame and one that is nowhere near any. One scale covers
# both halves of X; see openwiki/motion-matching/learned-motion-matching.md.
PROJECTOR_SIGMA = 1.0

# Candidate frames scored per matmul while searching for a training target, bounding peak GPU
# memory.
NEAREST_NEIGHBOUR_CHUNK = 16384

# Free-run lengths the stepper's drift is reported at, in database frames. Ten is the default
# search cadence, so the one that decides whether the stepper works; thirty shows whether it
# degrades or explodes past that.
DRIFT_HORIZONS = (5, 10, 20, 30)


@dataclass(frozen=True)
class LmmSpec:
    """
    Everything about the packing that a trained model has to be fed back exactly.

    Stored in the checkpoint and checked against the live database at load, because a
    training/inference disagreement here does not throw, it just makes the network wrong.

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
        nothing to be a matcher *of* without them -- see :func:`training.training_data.load_database`.
    """
    if training_set.features is None:
        raise ValueError(
            'this database has no .mmfeatures, so there is no query vector to learn against. '
            'Learned motion matching reads the same features the classic matcher searches; '
            'generate them from a MotionMatchingData asset that authors feature channels.')

    bone_indices = neural_packing.select_bones(
        training_set.bone_names, training_set.parents, excluded_bones)
    parents = fk.parents_within(training_set.parents, bone_indices)

    return LmmSpec(
        bone_indices=bone_indices,
        bone_names=tuple(training_set.bone_names[i] for i in bone_indices),
        parents=parents,
        rest_offsets=fk.rest_offsets(training_set.positions[0][bone_indices],
                                     training_set.rotations[0][bone_indices], parents),
        feature_size=int(training_set.features.shape[1]),
        latent_size=int(latent_size),
        frame_time=float(training_set.frame_time))


def pose_vector(training_set: TrainingSet, spec: LmmSpec) -> np.ndarray:
    """Every frame's ``Y``. (n_frames, :attr:`LmmSpec.pose_size`) float32."""
    n = training_set.n_frames
    bones = spec.bone_indices

    return np.concatenate([
        fk.local_rotations_6d(training_set.rotations[:, bones], spec.parents).reshape(n, -1),
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

    Read out of the training set, which is already in the character frame; :mod:`lmm.fk` derives
    ``Q`` from a *prediction*, and ``test_lmm_fk`` holds the two definitions to each other.
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

    Usability is :meth:`TrainingSet.usable`, which is feature validity and nothing else. Not
    interchangeable with :func:`pfnn.dataset.usable_queries`, whose gait-cycle filter would drop
    idling -- motion a matcher has to be able to produce.
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

    Pairs, because three terms of the autoencoder's loss are differences over adjacent frames.
    """
    latent_exists = compressible(training_set)

    next_exists = np.zeros(training_set.n_frames, dtype=bool)
    next_exists[:-1] = latent_exists[1:]

    return np.flatnonzero(latent_exists & next_exists).astype(np.int64)


def stepper_windows(training_set: TrainingSet,
                    window: int = DEFAULT_STEPPER_WINDOW) -> np.ndarray:
    """
    (m,) frame indices ``i`` where every frame of ``i .. i + window`` carries a latent.

    The starts of the unrolled runs the stepper trains on. **No window crosses a clip boundary**,
    which falls out of :func:`compressible`: a frame only has a latent when its successor is in the
    same clip. The stepper must never learn a jump -- the search is what jumps.

    :param window: frames the state is advanced over. See :data:`DEFAULT_STEPPER_WINDOW`.
    """
    return latent_runs(compressible(training_set), window)


def latent_runs(latent_exists: np.ndarray, window: int) -> np.ndarray:
    """
    (m,) indices ``i`` where ``latent_exists`` is true for all of ``i .. i + window``.

    Separate from :func:`stepper_windows` because rollout diagnostics need runs longer than the
    training window.

    :param latent_exists: (n,) as :func:`compressible` returns.
    :param window: how many further frames each run must cover.
    """
    if window < 1:
        raise ValueError(f'a stepper window has to be at least one frame, got {window}')

    runs = latent_exists.copy()
    for offset in range(1, window + 1):
        shifted = np.zeros_like(latent_exists)
        shifted[:-offset] = latent_exists[offset:]
        runs &= shifted

    return np.flatnonzero(runs).astype(np.int64)


def build_vectors(training_set: TrainingSet, spec: LmmSpec):
    """
    Pack the whole database once, as the arrays training indexes into.

    Returned per database frame rather than per sample, because the autoencoder, the latent bake
    and the stepper each index the same frames differently.

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


def state_scales(x: np.ndarray, latents: np.ndarray, latent_exists: np.ndarray):
    """
    ``(x_scale, z_scale)``: one scalar spread for each half of the state ``[X Z]``.

    One number per half rather than per float, for the reason :func:`group_normalization` gives.
    Both the stepper's loss and its drift are measured in these units, so they are defined once.
    """
    return (max(1e-6, float(x[latent_exists].std())),
            max(1e-6, float(latents[latent_exists].std())))


def group_normalization(vectors: np.ndarray, block_layout):
    """
    ``(mean, std)`` over packed vectors: a mean per float, but **one standard deviation per block**.

    The paper's scheme: dividing a 3-vector by three different numbers would warp it, so an L1
    error would mean something different per axis. A block that never varies is left unscaled.

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

    Must stay the same expansion ``MotionMatchingStage.UpdateFeatureWeights`` does, or the learned
    matcher approximates a search nobody runs.

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


def projector_noise_scale(x: np.ndarray, latent_exists: np.ndarray) -> np.ndarray:
    """
    (feature_size,) the spread each float of ``X`` is displaced by while fitting the projector.

    The projector has to answer queries genuinely off the manifold, not database rows with a little
    jitter. Each feature's own spread plus :data:`PROJECTOR_NOISE_FLOOR`, as in the released code.
    """
    return (x[latent_exists].std(axis=0) + PROJECTOR_NOISE_FLOOR).astype(np.float32)


def weighted_distance(a, b, weights):
    """
    Per-row distance between two batches of feature vectors, under the authored search weights.

    Euclidean rather than squared (``MotionMatchingStage.SqrDistance``) so it shares units with the
    L1 error on ``X`` it is summed beside; both agree on which candidate is nearer.
    """
    return torch.sqrt(torch.clamp((((a - b) ** 2) * weights).sum(dim=-1), min=1e-12))


def candidate_norms(candidates, weights):
    """
    The per-candidate constant :func:`nearest_neighbours` expands its distances around.

    Computed once per database and reused, which lets the search be a matmul.
    """
    return ((candidates ** 2) * weights).sum(dim=1)


def nearest_neighbours(queries, candidates, weights, norms,
                       chunk: int = NEAREST_NEIGHBOUR_CHUNK):
    """
    (n_queries,) the candidate nearest each query under the weighted metric.

    **The target must be the nearest neighbour of the noisy query**, not the frame the noise was
    added to; the latter silently trains a denoiser instead of the lookup. The weights are the
    authored ones for the same reason.

    :param queries: (n, feature_size) the displaced queries, in the database's own units.
    :param candidates: (m, feature_size) the frames the search may return.
    :param weights: (feature_size,) per-float search weights.
    :param norms: :func:`candidate_norms` over the same candidates and weights.
    :param chunk: candidates scored per matmul; see :data:`NEAREST_NEIGHBOUR_CHUNK`.
    """
    # The query's own squared norm is constant along a row, so it cannot change the argmin. Callers
    # measure the distance to the winner directly, avoiding this expansion's cancellation error.
    scaled = queries * weights
    best = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
    best_score = torch.full((queries.shape[0],), float('inf'), device=queries.device)

    for start in range(0, candidates.shape[0], chunk):
        stop = min(start + chunk, candidates.shape[0])
        scores = norms[start:stop] - 2.0 * (scaled @ candidates[start:stop].T)
        values, indices = scores.min(dim=1)

        taken = values < best_score
        best = torch.where(taken, indices + start, best)
        best_score = torch.where(taken, values, best_score)

    return best


def recall_against_search(queries, answered_x, answered_z, candidates, candidate_latents,
                          weights, norms, z_scale: float, chunk: int = 64) -> dict:
    """
    How a projector's answers compare with the lookup they replace, on the same queries.

    Three numbers, **none of which is sufficient alone** (see
    openwiki/motion-matching/learned-motion-matching.md):

    * **distance ratio** -- mean answer distance over mean true-nearest distance; 1.0 is the search
      itself, and below 1.0 means the answer lies between database states, not an improvement.
    * **top-1% recall** -- how often the answer is at least as near as the nearest one percent of
      frames; saturates at large displacements.
    * **latent error** -- how far the answer's latent is from the true neighbour's, which catches a
      plausible ``X`` carrying a latent from somewhere else.

    Shared by the trainer and the offline report so both measure the same thing.

    :param queries: (n, feature_size) the displaced queries that were answered.
    :param answered_x: (n, feature_size) the projector's feature vectors, in database units.
    :param answered_z: (n, latent_size) its latents.
    :param candidates: (m, feature_size) the frames a search could have returned.
    :param candidate_latents: (m, latent_size) their latents.
    :param norms: :func:`candidate_norms` over the same candidates and weights.
    :param chunk: queries per distance matrix; the whole database is a row of it.
    """
    if queries.shape[0] == 0:
        return {'distance_ratio': float('nan'), 'top_percent_recall': float('nan'),
                'latent_error': float('nan')}

    ratios, recalls, latent_errors = [], [], []
    kth = max(1, candidates.shape[0] // 100)

    with torch.no_grad():
        for start in range(0, queries.shape[0], chunk):
            batch = queries[start:start + chunk]

            # Unlike in `nearest_neighbours`, these distances are compared, not only ranked, so the
            # query's own norm is needed.
            query_norm = ((batch ** 2) * weights).sum(dim=1, keepdim=True)
            distances = torch.sqrt(torch.clamp(
                norms - 2.0 * ((batch * weights) @ candidates.T) + query_norm, min=0.0) + 1e-12)

            nearest = distances.argmin(dim=1)
            best = distances.gather(1, nearest.unsqueeze(1)).squeeze(1)
            threshold = distances.kthvalue(kth, dim=1).values

            answered = weighted_distance(batch, answered_x[start:start + chunk], weights)
            ratios.append(torch.stack([answered.sum(), best.sum()]))
            recalls.append((answered <= threshold).float())
            latent_errors.append(
                (answered_z[start:start + chunk] - candidate_latents[nearest])
                .abs().mean(dim=1) / z_scale)

    totals = torch.stack(ratios).sum(dim=0)
    return {
        'distance_ratio': float(totals[0] / torch.clamp(totals[1], min=1e-6)),
        'top_percent_recall': float(torch.cat(recalls).mean()),
        'latent_error': float(torch.cat(latent_errors).mean()),
    }
