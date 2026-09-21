"""
The character frame a pose database is matched in, derived from the stored poses.

Poses are stored exactly as the clips define them: bone 0 carries the rig root's world
position and rotation, bones 1+ carry rest offsets and parent-local rotations. The
ground-projected, yaw-only frame the motion field is defined in is not stored -- it is
reconstructed here from a reference bone and a bone-local "forward" axis. The definition
is structural, not authored: the reference bone is joint 0 and the forward axis is the
one that points along character forward in joint 0's rest rotation. Deriving it keeps one
skeleton for the clips, the database and the rig, and makes the frame exactly reproducible
on the C# side (``SimulationFrame``), which builds the same definition off the skeleton.

Conventions, all shared with the rest of this package:

* Quaternions are xyzw, matching the packed pose layout and
  :class:`scipy.spatial.transform.Rotation`.
* Unity is y-up and left-handed, so the character faces +z and the frame's yaw is
  ``atan2(forward.x, forward.z)``.
* Angular rates are rotation vectors (axis * radians per second), left-multiplied:
  ``next = exp(rate * dt) * current``, taken the short way round (w >= 0).
* Velocities are per-second rates. One frame's delta is ``rate * frame_time``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from core.pose_set import PoseSet
from core.skeleton import Skeleton

# Heading used when the reference bone's forward axis projects to nothing on the ground
# -- looking straight up or down. Matches Unity's LookRotation falling back to identity.
_DEGENERATE_FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float64)


class DerivedFrames(NamedTuple):
    """
    A pose sequence's character frames and the root state measured inside them.

    :param position: (n, 3) frame origin in world space, on the ground plane.
    :param yaw: (n,) frame heading in radians.
    :param rotation: (n, 4) frame rotation, a pure yaw quaternion.
    :param root_position: (n, 3) root bone position in frame space.
    :param root_rotation: (n, 4) root bone rotation in frame space, w >= 0.
    """

    position: np.ndarray
    yaw: np.ndarray
    rotation: np.ndarray
    root_position: np.ndarray
    root_rotation: np.ndarray


class FrameRates(NamedTuple):
    """
    Per-second rates of the quantities in :class:`DerivedFrames`.

    :param linear: (n, 3) frame origin velocity, in the frame's own space.
    :param yaw: (n,) frame heading rate in radians per second.
    :param root_linear: (n, 3) root position rate, in frame space.
    :param root_angular: (n, 3) root rotation rate in frame space, as a rotation vector.
    """

    linear: np.ndarray
    yaw: np.ndarray
    root_linear: np.ndarray
    root_angular: np.ndarray


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap radians into [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def canonical_quaternions(quats: np.ndarray) -> np.ndarray:
    """
    Flip quaternions onto the w >= 0 hemisphere.

    q and -q are the same rotation but only one is the short way round, so a difference
    of two rotations comes back as 10 degrees rather than 350. Mirrors C#
    ``MathExtensions.Abs``.
    """
    quats = np.asarray(quats)
    flip = quats[..., 3] < 0.0
    return np.where(flip[..., np.newaxis], -quats, quats)


def parent_indices(skeleton: Skeleton) -> np.ndarray:
    """
    Parent index per joint in the skeleton's iteration order, -1 for the root.

    The serialized bone order and this depth-first walk are the same order -- the whole
    package indexes pose arrays with joint positions from ``list(skeleton)``.
    """
    joints = list(skeleton)
    index_of = {joint: i for i, joint in enumerate(joints)}
    return np.array(
        [-1 if joint.parent() is None else index_of[joint.parent()] for joint in joints],
        dtype=np.int64)


def bone_world_transform(positions: np.ndarray,
                         rotations: np.ndarray,
                         parents: np.ndarray,
                         bone_index: int) -> tuple[np.ndarray, Rotation]:
    """
    World position and rotation of one bone, for a whole sequence at once.

    :param positions: (n, num_bones, 3) parent-local joint positions; slot 0 is the
        root's world position.
    :param rotations: (n, num_bones, 4) parent-local joint rotations, xyzw; slot 0 is
        the root's world rotation.
    :param parents: (num_bones,) parent index per bone, -1 for the root.
    :param bone_index: bone to resolve.
    :return: ``(position, rotation)``, shapes (n, 3) and a length-n ``Rotation``.
    """
    chain = []
    index = int(bone_index)
    while index >= 0:
        chain.append(index)
        index = int(parents[index])
    chain.reverse()

    position = np.asarray(positions[:, chain[0], :], dtype=np.float64)
    rotation = Rotation.from_quat(rotations[:, chain[0], :])
    for index in chain[1:]:
        position = position + rotation.apply(positions[:, index, :])
        rotation = rotation * Rotation.from_quat(rotations[:, index, :])

    return position, rotation


def forward_kinematics(positions: np.ndarray,
                       rotations: np.ndarray,
                       parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    World position and rotation of *every* bone, for a whole sequence at once.

    :func:`bone_world_transform` walks the chain up from one bone, which is the right shape
    when a caller wants the reference bone and nothing else. This is the other case: every
    bone, where walking each chain separately would redo the shared prefix once per leaf.
    Bones are stored in depth-first order, so a single forward pass sees every parent before
    its children.

    :param positions: (n, num_bones, 3) parent-local joint positions; slot 0 is the root's
        world position.
    :param rotations: (n, num_bones, 4) parent-local joint rotations, xyzw; slot 0 is the
        root's world rotation.
    :param parents: (num_bones,) parent index per bone, -1 for the root.
    :return: ``(positions, rotations)``, shapes (n, num_bones, 3) and (n, num_bones, 4).
    """
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    n_frames, n_bones = positions.shape[0], positions.shape[1]

    world_positions = np.zeros((n_frames, n_bones, 3), dtype=np.float64)
    world_rotations = np.zeros((n_frames, n_bones, 4), dtype=np.float64)
    world_positions[:, 0] = positions[:, 0]
    world_rotations[:, 0] = rotations[:, 0]

    for bone in range(1, n_bones):
        parent = int(parents[bone])
        parent_rotation = Rotation.from_quat(world_rotations[:, parent])
        world_positions[:, bone] = (world_positions[:, parent] +
                                    parent_rotation.apply(positions[:, bone]))
        world_rotations[:, bone] = (parent_rotation *
                                    Rotation.from_quat(rotations[:, bone])).as_quat()

    return world_positions, world_rotations


def yaw_quaternion(yaw: np.ndarray) -> np.ndarray:
    """Pure yaw quaternions from headings in radians, xyzw."""
    yaw = np.asarray(yaw, dtype=np.float64)
    half = 0.5 * yaw
    quats = np.zeros(yaw.shape + (4,), dtype=np.float64)
    quats[..., 1] = np.sin(half)
    quats[..., 3] = np.cos(half)
    return quats


def frame_yaw(rotations: Rotation, forward_local: np.ndarray) -> np.ndarray:
    """
    Heading of a bone in radians: its forward axis flattened onto the ground plane.

    :param rotations: length-n world rotations of the reference bone.
    :param forward_local: (3,) reference-bone-local axis that means "character forward".
    """
    forward = np.asarray(rotations.apply(np.asarray(forward_local, dtype=np.float64)),
                         dtype=np.float64)
    forward[..., 1] = 0.0

    lengths = np.linalg.norm(forward, axis=-1, keepdims=True)
    forward = np.where(lengths > 1e-6, forward / np.maximum(lengths, 1e-12),
                       _DEGENERATE_FORWARD)

    return np.arctan2(forward[..., 0], forward[..., 2])


def derive_frames(positions: np.ndarray,
                  rotations: np.ndarray,
                  parents: np.ndarray,
                  reference_bone_index: int,
                  forward_local: np.ndarray) -> DerivedFrames:
    """
    Character frames for a pose sequence, plus the root state measured inside them.

    The frame sits under the reference bone on the ground and faces where that bone
    faces, so expressing the pose in it removes exactly the translation and yaw the
    motion field must not see. Applying this to an already frame-local sequence is the
    identity: the reference bone's own horizontal offset and yaw are zero by
    construction.

    :param positions: (n, num_bones, 3) joint positions, slot 0 world.
    :param rotations: (n, num_bones, 4) joint rotations xyzw, slot 0 world.
    :param parents: (num_bones,) parent index per bone, -1 for the root.
    :param reference_bone_index: bone the frame is derived from.
    :param forward_local: (3,) forward axis in that bone's local space.
    """
    reference_position, reference_rotation = bone_world_transform(
        positions, rotations, parents, reference_bone_index)

    frame_position = reference_position.copy()
    frame_position[:, 1] = 0.0

    yaw = frame_yaw(reference_rotation, forward_local)
    frame_rotation = yaw_quaternion(yaw)
    inverse_frame = Rotation.from_quat(frame_rotation).inv()

    root_position = inverse_frame.apply(
        np.asarray(positions[:, 0, :], dtype=np.float64) - frame_position)
    root_rotation = canonical_quaternions(
        (inverse_frame * Rotation.from_quat(rotations[:, 0, :])).as_quat())

    return DerivedFrames(
        position=frame_position.astype(np.float32),
        yaw=yaw.astype(np.float32),
        rotation=frame_rotation.astype(np.float32),
        root_position=np.asarray(root_position, dtype=np.float32),
        root_rotation=np.asarray(root_rotation, dtype=np.float32))


def frame_rates(frames: DerivedFrames, frame_time: float) -> FrameRates:
    """
    Finite-difference the derived sequences into per-second rates.

    Returns one rate fewer than there are frames: rate ``i`` carries frame ``i`` onto
    frame ``i + 1``, so the caller must not feed it frames from two different clips.
    """
    inverse_frame_time = 1.0 / float(frame_time)

    inverse_rotation = Rotation.from_quat(frames.rotation[:-1]).inv()
    linear = inverse_rotation.apply(
        np.asarray(frames.position[1:] - frames.position[:-1], dtype=np.float64))
    linear = np.asarray(linear) * inverse_frame_time

    yaw = wrap_to_pi(np.asarray(frames.yaw[1:], dtype=np.float64) -
                     frames.yaw[:-1]) * inverse_frame_time

    root_linear = (np.asarray(frames.root_position[1:], dtype=np.float64) -
                   frames.root_position[:-1]) * inverse_frame_time

    delta = (Rotation.from_quat(frames.root_rotation[1:]) *
             Rotation.from_quat(frames.root_rotation[:-1]).inv())
    root_angular = Rotation.from_quat(
        canonical_quaternions(delta.as_quat())).as_rotvec() * inverse_frame_time

    return FrameRates(linear=linear.astype(np.float32),
                      yaw=yaw.astype(np.float32),
                      root_linear=root_linear.astype(np.float32),
                      root_angular=np.asarray(root_angular, dtype=np.float32))


def clip_ranges(clips, n_poses: int) -> list[tuple[int, int]]:
    """Clip frame ranges clamped to the arrays actually read, empty ones dropped."""
    if not clips:
        return [(0, n_poses)] if n_poses > 0 else []

    ranges = [(int(clip['start']), min(int(clip['end']), n_poses)) for clip in clips]
    return [(start, end) for start, end in ranges if end > start]


def extend_by_one_frame(positions: np.ndarray,
                         rotations: np.ndarray,
                         velocities: np.ndarray,
                         angular_velocities: np.ndarray,
                         frame_time: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-create the frame just past the end of a clip from its last stored velocities.

    Every stored pose has a velocity, including the last of a clip: the extractor
    samples one frame beyond what it stores and differences against it. Stepping the
    last pose by its own stored rates inverts that exactly, which is what lets a
    derived rate exist for every stored frame rather than for all but the last.
    """
    last_position = positions[-1] + velocities[-1] * frame_time
    last_rotation = (Rotation.from_rotvec(angular_velocities[-1] * frame_time) *
                     Rotation.from_quat(rotations[-1])).as_quat()

    return (np.concatenate([positions, last_position[np.newaxis, ...]], axis=0),
            np.concatenate([rotations, last_rotation[np.newaxis, ...]], axis=0))


def derive_pose_set_frames(pose_set: PoseSet) -> tuple[DerivedFrames, FrameRates]:
    """
    Derive the character frame of every pose in a database, clip by clip.

    Both results are indexed by global pose index, and the rates are aligned with the
    poses rather than lagging them -- rate ``i`` still describes the step out of pose
    ``i``, and at the end of a clip that step lands on the frame reconstructed by
    :func:`extend_by_one_frame` instead of on the unrelated first pose of the next
    clip.

    :param pose_set: a database read by ``formats.pose_set_importer``, which is where the
        reference bone and forward axis come from.
    """
    positions = pose_set.local_positions
    rotations = pose_set.local_rotations
    velocities = pose_set.local_velocities
    angular_velocities = pose_set.local_angular_velocities
    frame_time = float(pose_set.frameTime)

    parents = parent_indices(pose_set.skeleton)
    reference_bone_index = int(pose_set.sim_frame_bone_index)
    forward_local = pose_set.sim_frame_forward

    n_poses = positions.shape[0]
    frames = DerivedFrames(position=np.zeros((n_poses, 3), dtype=np.float32),
                           yaw=np.zeros(n_poses, dtype=np.float32),
                           rotation=np.zeros((n_poses, 4), dtype=np.float32),
                           root_position=np.zeros((n_poses, 3), dtype=np.float32),
                           root_rotation=np.zeros((n_poses, 4), dtype=np.float32))
    rates = FrameRates(linear=np.zeros((n_poses, 3), dtype=np.float32),
                       yaw=np.zeros(n_poses, dtype=np.float32),
                       root_linear=np.zeros((n_poses, 3), dtype=np.float32),
                       root_angular=np.zeros((n_poses, 3), dtype=np.float32))

    for start, end in clip_ranges(pose_set.clips, n_poses):
        clip_positions, clip_rotations = extend_by_one_frame(
            positions[start:end], rotations[start:end],
            velocities[start:end], angular_velocities[start:end], frame_time)

        clip_frames = derive_frames(clip_positions, clip_rotations, parents,
                                    reference_bone_index, forward_local)
        clip_rates = frame_rates(clip_frames, frame_time)

        for field, clip_field in zip(frames, clip_frames):
            field[start:end] = clip_field[:-1]
        for field, clip_field in zip(rates, clip_rates):
            field[start:end] = clip_field

    return frames, rates
