"""
Turns a Unity pose database into the per-frame arrays a neural motion model trains on.

Both of the models this repository is heading towards want the same handful of quantities
per frame, expressed in the character frame rather than the world:

* **PFNN** takes a trajectory window, the previous frame's joint positions and velocities,
  and a gait phase; it predicts the next pose, the root delta and a phase increment.
* **Learned motion matching** takes the matching feature vector as its query and trains a
  decompressor to reconstruct a full pose from a latent, so its target is the pose itself:
  joint positions, rotations, velocities and angular velocities, plus the root's motion.

So this module does not build either network's input tensor. It builds the shared thing
underneath both -- every frame of the database, in the frame the character is measured in,
with the derived quantities that are not stored anywhere -- and leaves the packing to
whoever is training. :meth:`TrainingSet.pose_vector` offers one such packing for the
decompressor target, and :meth:`TrainingSet.trajectory_window` samples a trajectory at
whatever offsets a model wants; both are conveniences rather than formats.

What is *derived* here, and why it cannot simply be read out of the ``.mmpose``:

* Poses are stored parent-local with bone 0 in world space, so joint positions in the
  character frame need forward kinematics and then the frame transform.
* The stored per-bone velocities are parent-local channel differences, which is not the
  same quantity as a joint's velocity within the character frame.
* Gait phase is not stored at all; it is reconstructed from the foot contacts by
  :mod:`gait_phase`.
* A trajectory window is where the character was and will be relative to now, which the
  frame transform is precisely what removes -- so it is derived from the world frames,
  which are kept for exactly that.

Everything is computed **clip by clip**. Clips sit back to back in one array, so
differencing across a boundary would report a jump cut as motion. The last frame of each
clip is the one case that needs care: it has no successor in the array, and
:func:`simulation_frame.extend_by_one_frame` reconstructs the frame just past it from the
stored velocities, which is exactly what the extractor differenced against when it wrote
them.

Conventions are the package's own: quaternions xyzw, y-up left-handed with the character
facing +z, velocities as per-second rates.

The C# counterpart of the frame-local conversion is ``CharacterSpacePose``, which is what a stage
running a trained model feeds it at inference. Keeping the two in agreement is the point of having
one named definition on each side: a mismatch in frame, units or rate convention does not throw, it
just makes the network wrong. The one place the two are not identical by construction is the rates --
here they are differences of consecutive frame-local poses, there the instantaneous rate implied by a
pose's own velocity channels -- and since those channels are themselves finite differences over the
same timestep, the two agree to first order and exactly for motion that is rigid within the frame.

Run it as a script to write an ``.npz`` beside a database::

    python training_data.py Assets/StreamingAssets/MMDatabases/MotionMatchingData \\
        MotionMatchingData --out training.npz
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

import gait_phase
from Animation import PoseSet
from feature_set_importer import FeatureSet, read_feature_set
from pose_set_importer import deserialize_pose_set
from simulation_frame import (canonical_quaternions, clip_ranges, derive_frames,
                              extend_by_one_frame, forward_kinematics, frame_rates,
                              parent_indices)


def rotations_to_6d(rotations: np.ndarray) -> np.ndarray:
    """
    The two-axis rotation representation a network should regress instead of a quaternion.

    A quaternion is a double cover of SO(3) and its unit constraint is not something a
    linear output layer can respect, so regressing one gives a discontinuous target that
    has to be renormalised afterwards. The first two columns of the rotation matrix are
    continuous and unconstrained, and the third column is their cross product, so nothing
    is lost. (Zhou et al., *On the Continuity of Rotation Representations in Neural
    Networks*.)

    :param rotations: (..., 4) quaternions, xyzw.
    :return: (..., 6) float32, the two columns concatenated.
    """
    rotations = np.asarray(rotations)
    flat = rotations.reshape(-1, 4)
    matrices = Rotation.from_quat(flat).as_matrix()
    return matrices[:, :, :2].transpose(0, 2, 1).reshape(rotations.shape[:-1] + (6,)) \
        .astype(np.float32)


@dataclass
class TrainingSet:
    """
    Every frame of a pose database, in the character frame, with what a model needs added.

    All per-frame arrays are indexed by global frame, matching the pose database, and
    :attr:`clip_ranges` is what maps a global index back to its own animation. A sampler
    must never take a window that crosses one of those boundaries.

    :param frame_time: seconds per frame. Every rate below is per second, so a single
        frame's delta is ``rate * frame_time``.
    :param clip_ranges: (n_clips, 2) int32 half-open ``[start, end)`` frame ranges.
    :param bone_names: one name per bone, in the database's depth-first bone order.
    :param parents: (n_bones,) int32 parent index per bone, -1 for the root.
    :param positions: (n, n_bones, 3) joint positions in the character frame.
    :param rotations: (n, n_bones, 4) joint rotations in the character frame, xyzw, w >= 0.
    :param velocities: (n, n_bones, 3) rate of change of :attr:`positions`, m/s. This is
        motion *relative to the moving character frame*, which is what a network predicting
        a pose should see -- the frame's own travel is :attr:`root_velocity`.
    :param angular_velocities: (n, n_bones, 3) rate of change of :attr:`rotations` as a
        rotation vector, rad/s, taken the short way round.
    :param root_velocity: (n, 3) velocity of the character frame's origin, in its own
        space, m/s.
    :param root_yaw_rate: (n,) rate of change of the character frame's heading, rad/s.
    :param contacts: (n, 2) float32 foot contact flags, left then right.
    :param phase: (n,) gait phase in [0, 2*pi), from :mod:`gait_phase`.
    :param phase_rate: (n,) rate of change of the unwrapped phase, rad/s. Zero marks a
        clip with no measurable gait cycle.
    :param frame_position: (n, 3) **world** position of the character frame's origin, on the
        ground plane. Not model input -- see the note below.
    :param frame_yaw: (n,) **world** heading of the character frame, in radians. Likewise.
    :param features: (n, feature_size) normalised matching feature vectors, or None when
        no ``.mmfeatures`` was read. This is the query a learned motion matching projector
        maps into.
    :param feature_valid: (n,) bool, False where the feature vector is not usable.
    :param feature_mean: (feature_size,) normalisation mean.
    :param feature_std: (feature_size,) normalisation standard deviation.
    :param feature_names: one name per feature, in vector order.
    :param feature_widths: (n_features,) floats per sample of each feature.
    :param feature_counts: (n_features,) samples stored per feature.
    :param n_trajectory_features: how many of the above are trajectory features; the rest
        are pose features, and their first float offset is where the pose block starts.

    :attr:`frame_position` and :attr:`frame_yaw` are the only world-space arrays here, and a
    model must not take either: where the character stands and which way it faces are exactly
    what a locomotion model has to be invariant to. They are kept because a trajectory window
    is where the character was and will be *relative to now*, which cannot be reconstructed
    from frame-local data -- that information is precisely what the frame transform removed.
    See :meth:`trajectory_window`.
    """

    frame_time: float
    clip_ranges: np.ndarray
    bone_names: list[str]
    parents: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray
    root_velocity: np.ndarray
    root_yaw_rate: np.ndarray
    contacts: np.ndarray
    phase: np.ndarray
    phase_rate: np.ndarray
    frame_position: np.ndarray
    frame_yaw: np.ndarray

    features: np.ndarray | None = None
    feature_valid: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    feature_names: list[str] | None = None
    feature_widths: np.ndarray | None = None
    feature_counts: np.ndarray | None = None
    n_trajectory_features: int = 0

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_bones(self) -> int:
        return int(self.positions.shape[1])

    @property
    def rotations_6d(self) -> np.ndarray:
        """:attr:`rotations` in the two-axis form -- see :func:`rotations_to_6d`."""
        return rotations_to_6d(self.rotations)

    def pose_vector(self) -> np.ndarray:
        """
        One canonical flat packing of the pose, for a decompressor target.

        ``[positions | rotations_6d | velocities | angular_velocities | root_velocity |
        root_yaw_rate | contacts]``, so ``15 * n_bones + 6`` floats per frame. Named by
        :meth:`pose_vector_layout`.

        This is a convenience, not a format: nothing reads it back, and a caller with a
        different target in mind should pack the arrays directly.
        """
        n = self.n_frames
        return np.concatenate([
            self.positions.reshape(n, -1),
            self.rotations_6d.reshape(n, -1),
            self.velocities.reshape(n, -1),
            self.angular_velocities.reshape(n, -1),
            self.root_velocity,
            self.root_yaw_rate.reshape(n, 1),
            self.contacts,
        ], axis=1).astype(np.float32)

    def pose_vector_layout(self) -> list[tuple[str, int, int]]:
        """``(name, offset, float_count)`` per block of :meth:`pose_vector`."""
        blocks = [("positions", 3 * self.n_bones),
                  ("rotations_6d", 6 * self.n_bones),
                  ("velocities", 3 * self.n_bones),
                  ("angular_velocities", 3 * self.n_bones),
                  ("root_velocity", 3),
                  ("root_yaw_rate", 1),
                  ("contacts", 2)]

        layout = []
        offset = 0
        for name, count in blocks:
            layout.append((name, offset, count))
            offset += count
        return layout

    def trajectory_window(self, offsets) -> tuple[np.ndarray, np.ndarray]:
        """
        Where the character was and will be, around every frame, in that frame's own space.

        This is the input a phase-functioned network is organised around, and the one thing
        a model needs that cannot be recovered from the frame-local arrays: they have had
        exactly this information removed. Motion matching's trajectory features are the same
        idea, but only for the horizons a ``MotionMatchingData`` happens to author, whereas a
        PFNN-style window is a dense sweep of roughly a second either side.

        Offsets are clamped to the containing clip rather than wrapped or dropped, so a
        window near a clip edge repeats its last real sample. That is the same thing a
        character standing still would produce, and it keeps every frame usable instead of
        discarding the ends of every clip.

        :param offsets: frame offsets relative to the query frame, negative for the past.
        :return: ``(positions, directions)``, both (n_frames, len(offsets), 2) float32 in the
            ground plane as (x, z). Positions are metres from the query frame's origin;
            directions are unit facings.
        """
        offsets = np.asarray(offsets, dtype=np.int64)
        n_frames, n_offsets = self.n_frames, offsets.size

        # Clamp each offset into the clip that owns the query frame.
        starts = np.zeros(n_frames, dtype=np.int64)
        ends = np.zeros(n_frames, dtype=np.int64)
        for start, end in self.clip_ranges:
            starts[start:end] = start
            ends[start:end] = end - 1

        sampled = np.arange(n_frames, dtype=np.int64)[:, np.newaxis] + offsets[np.newaxis, :]
        sampled = np.clip(sampled, starts[:, np.newaxis], ends[:, np.newaxis])

        origin = np.asarray(self.frame_position, dtype=np.float64)[:, [0, 2]]
        yaw = np.asarray(self.frame_yaw, dtype=np.float64)

        # Rotating a world offset into the query frame is the inverse yaw. Unity is y-up and
        # left-handed with the character facing +z, so a heading of theta is the direction
        # (sin theta, cos theta) in (x, z) and the inverse rotation is its transpose.
        cos, sin = np.cos(yaw)[:, np.newaxis], np.sin(yaw)[:, np.newaxis]

        offset_world = origin[sampled] - origin[:, np.newaxis, :]
        positions = np.stack([
            cos * offset_world[..., 0] - sin * offset_world[..., 1],
            sin * offset_world[..., 0] + cos * offset_world[..., 1],
        ], axis=-1)

        relative_yaw = yaw[sampled] - yaw[:, np.newaxis]
        directions = np.stack([np.sin(relative_yaw), np.cos(relative_yaw)], axis=-1)

        return positions.astype(np.float32), directions.astype(np.float32)

    def usable(self) -> np.ndarray:
        """
        (n,) bool: frames a model may train on.

        A frame is usable when its matching feature vector is valid, which is the stricter
        of the two conditions -- Unity already excludes the frames whose trajectory window
        would leave their own clip. With no feature set read, every frame qualifies.
        """
        if self.feature_valid is None:
            return np.ones(self.n_frames, dtype=bool)
        return self.feature_valid.astype(bool)


def _character_space(world_positions: np.ndarray,
                     world_rotations: np.ndarray,
                     frame_positions: np.ndarray,
                     frame_rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Move every joint of every frame into that frame's own character frame.

    :param world_positions: (n, n_bones, 3)
    :param world_rotations: (n, n_bones, 4) xyzw
    :param frame_positions: (n, 3) character-frame origins
    :param frame_rotations: (n, 4) character-frame rotations, pure yaw
    """
    n_frames, n_bones = world_positions.shape[0], world_positions.shape[1]

    # scipy takes flat stacks of rotations, so the frame rotation is repeated once per bone
    # and both arrays are flattened to (n_frames * n_bones, ...).
    inverse_frame = Rotation.from_quat(np.repeat(frame_rotations, n_bones, axis=0)).inv()

    offsets = (world_positions - frame_positions[:, np.newaxis, :]).reshape(-1, 3)
    positions = inverse_frame.apply(offsets).reshape(n_frames, n_bones, 3)

    rotations = (inverse_frame * Rotation.from_quat(world_rotations.reshape(-1, 4))).as_quat()
    rotations = canonical_quaternions(rotations).reshape(n_frames, n_bones, 4)

    return positions, rotations


def _joint_rates(positions: np.ndarray, rotations: np.ndarray, frame_time: float) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference character-space joints into per-second rates.

    Takes ``n + 1`` frames and returns ``n`` rates: rate ``i`` carries frame ``i`` onto
    frame ``i + 1``, the same alignment :func:`simulation_frame.frame_rates` uses.
    """
    n_frames, n_bones = positions.shape[0] - 1, positions.shape[1]
    inverse_frame_time = 1.0 / float(frame_time)

    linear = (positions[1:] - positions[:-1]) * inverse_frame_time

    delta = (Rotation.from_quat(rotations[1:].reshape(-1, 4)) *
             Rotation.from_quat(rotations[:-1].reshape(-1, 4)).inv())
    angular = Rotation.from_quat(canonical_quaternions(delta.as_quat())).as_rotvec()
    angular = angular.reshape(n_frames, n_bones, 3) * inverse_frame_time

    return linear, angular


def build_training_set(pose_set: PoseSet, feature_set: FeatureSet | None = None) -> TrainingSet:
    """
    Assemble a :class:`TrainingSet` from a database read by ``pose_set_importer``.

    :param pose_set: the pose database, carrying the simulation-frame definition
        ``pose_set_importer`` attached to it.
    :param feature_set: the matching feature database, when one is available. Must have one
        vector per pose.
    :raises ValueError: the two databases disagree about how many frames there are.
    """
    positions = np.asarray(pose_set.local_positions)
    rotations = np.asarray(pose_set.local_rotations)
    velocities = np.asarray(pose_set.local_velocities)
    angular_velocities = np.asarray(pose_set.local_angular_velocities)
    frame_time = float(pose_set.frameTime)
    n_frames, n_bones = positions.shape[0], positions.shape[1]

    if feature_set is not None and feature_set.n_frames != n_frames:
        raise ValueError(
            f"The feature database holds {feature_set.n_frames} vectors but the pose database "
            f"holds {n_frames} poses. Regenerate both from the same asset.")

    parents = parent_indices(pose_set.skeleton)
    reference_bone = int(pose_set.sim_frame_bone_index)
    forward_local = pose_set.sim_frame_forward

    out_positions = np.zeros((n_frames, n_bones, 3), dtype=np.float32)
    out_rotations = np.zeros((n_frames, n_bones, 4), dtype=np.float32)
    out_velocities = np.zeros((n_frames, n_bones, 3), dtype=np.float32)
    out_angular = np.zeros((n_frames, n_bones, 3), dtype=np.float32)
    out_root_velocity = np.zeros((n_frames, 3), dtype=np.float32)
    out_root_yaw_rate = np.zeros(n_frames, dtype=np.float32)
    out_frame_position = np.zeros((n_frames, 3), dtype=np.float32)
    out_frame_yaw = np.zeros(n_frames, dtype=np.float32)

    ranges = clip_ranges(pose_set.clips, n_frames)
    for start, end in ranges:
        # One frame past the clip, reconstructed from the stored velocities, so the last
        # real frame has a successor to difference against that is not the next animation.
        clip_positions, clip_rotations = extend_by_one_frame(
            positions[start:end], rotations[start:end],
            velocities[start:end], angular_velocities[start:end], frame_time)

        world_positions, world_rotations = forward_kinematics(
            clip_positions, clip_rotations, parents)
        frames = derive_frames(clip_positions, clip_rotations, parents,
                               reference_bone, forward_local)

        local_positions, local_rotations = _character_space(
            world_positions, world_rotations, frames.position, frames.rotation)
        linear, angular = _joint_rates(local_positions, local_rotations, frame_time)
        rates = frame_rates(frames, frame_time)

        out_positions[start:end] = local_positions[:-1]
        out_rotations[start:end] = local_rotations[:-1]
        out_velocities[start:end] = linear
        out_angular[start:end] = angular
        out_root_velocity[start:end] = rates.linear
        out_root_yaw_rate[start:end] = rates.yaw
        out_frame_position[start:end] = frames.position[:-1]
        out_frame_yaw[start:end] = frames.yaw[:-1]

    contacts = np.asarray(pose_set.foot_contacts).astype(np.float32)
    phase, phase_rate = gait_phase.pose_set_phase(contacts, ranges, frame_time)

    training_set = TrainingSet(
        frame_time=frame_time,
        clip_ranges=np.asarray(ranges, dtype=np.int32).reshape(-1, 2),
        bone_names=[joint.name for joint in pose_set.skeleton],
        parents=parents.astype(np.int32),
        positions=out_positions,
        rotations=out_rotations,
        velocities=out_velocities,
        angular_velocities=out_angular,
        root_velocity=out_root_velocity,
        root_yaw_rate=out_root_yaw_rate,
        contacts=contacts,
        phase=phase,
        phase_rate=phase_rate,
        frame_position=out_frame_position,
        frame_yaw=out_frame_yaw)

    if feature_set is not None:
        training_set.features = feature_set.features
        training_set.feature_valid = feature_set.valid
        training_set.feature_mean = feature_set.mean
        training_set.feature_std = feature_set.std
        training_set.feature_names = [entry.name for entry in feature_set.schema]
        training_set.feature_widths = np.array(
            [entry.floats_per_prediction for entry in feature_set.schema], dtype=np.int32)
        training_set.feature_counts = np.array(
            [entry.prediction_count for entry in feature_set.schema], dtype=np.int32)
        training_set.n_trajectory_features = sum(
            1 for entry in feature_set.schema if entry.is_trajectory)

    return training_set


def load_database(path: str, name: str, with_features: bool = True) -> TrainingSet:
    """
    Read a generated database from ``StreamingAssets`` and build a training set from it.

    :param path: the database folder, e.g.
        ``Assets/StreamingAssets/MMDatabases/MotionMatchingData``.
    :param name: the base file name, which is the Unity asset's name.
    :param with_features: read the ``.mmfeatures`` beside the poses. A ``MotionFieldConfig``
        database has none, so pass False for one of those.
    """
    pose_set = deserialize_pose_set(path, name)
    feature_set = read_feature_set(path, name) if with_features else None
    return build_training_set(pose_set, feature_set)


def save_npz(training_set: TrainingSet, path: str) -> None:
    """
    Write a training set as a single ``.npz``.

    Field names match the :class:`TrainingSet` attributes, so a consumer that never imports
    this package can still read it with :func:`numpy.load` and the class docstring.
    """
    arrays = {
        'frame_time': np.float32(training_set.frame_time),
        'clip_ranges': training_set.clip_ranges,
        'bone_names': np.array(training_set.bone_names, dtype=object),
        'parents': training_set.parents,
        'positions': training_set.positions,
        'rotations': training_set.rotations,
        'velocities': training_set.velocities,
        'angular_velocities': training_set.angular_velocities,
        'root_velocity': training_set.root_velocity,
        'root_yaw_rate': training_set.root_yaw_rate,
        'contacts': training_set.contacts,
        'phase': training_set.phase,
        'phase_rate': training_set.phase_rate,
        'frame_position': training_set.frame_position,
        'frame_yaw': training_set.frame_yaw,
    }

    if training_set.features is not None:
        arrays.update({
            'features': training_set.features,
            'feature_valid': training_set.feature_valid,
            'feature_mean': training_set.feature_mean,
            'feature_std': training_set.feature_std,
            'feature_names': np.array(training_set.feature_names, dtype=object),
            'feature_widths': training_set.feature_widths,
            'feature_counts': training_set.feature_counts,
            'n_trajectory_features': np.int32(training_set.n_trajectory_features),
        })

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> TrainingSet:
    """Read back what :func:`save_npz` wrote."""
    with np.load(path, allow_pickle=True) as data:
        training_set = TrainingSet(
            frame_time=float(data['frame_time']),
            clip_ranges=data['clip_ranges'],
            bone_names=[str(name) for name in data['bone_names']],
            parents=data['parents'],
            positions=data['positions'],
            rotations=data['rotations'],
            velocities=data['velocities'],
            angular_velocities=data['angular_velocities'],
            root_velocity=data['root_velocity'],
            root_yaw_rate=data['root_yaw_rate'],
            contacts=data['contacts'],
            phase=data['phase'],
            phase_rate=data['phase_rate'],
            frame_position=data['frame_position'],
            frame_yaw=data['frame_yaw'])

        if 'features' in data:
            training_set.features = data['features']
            training_set.feature_valid = data['feature_valid']
            training_set.feature_mean = data['feature_mean']
            training_set.feature_std = data['feature_std']
            training_set.feature_names = [str(name) for name in data['feature_names']]
            training_set.feature_widths = data['feature_widths']
            training_set.feature_counts = data['feature_counts']
            training_set.n_trajectory_features = int(data['n_trajectory_features'])

    return training_set


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('database', help='the generated database folder under StreamingAssets')
    parser.add_argument('name', help='base file name, i.e. the Unity asset name')
    parser.add_argument('--out', default=None, help='where to write the .npz (default: <name>.training.npz)')
    parser.add_argument('--no-features', action='store_true',
                        help='skip the .mmfeatures, for a database that has none')
    args = parser.parse_args()

    training_set = load_database(args.database, args.name, with_features=not args.no_features)
    out = args.out or os.path.join(args.database, f'{args.name}.training.npz')
    save_npz(training_set, out)

    usable = int(training_set.usable().sum())
    print(f"{training_set.n_frames} frames ({usable} usable) over {training_set.n_bones} bones "
          f"in {len(training_set.clip_ranges)} clips -> {out}")


if __name__ == '__main__':
    _main()
