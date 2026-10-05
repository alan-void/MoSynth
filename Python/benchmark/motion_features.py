"""
Per-frame motion features measured the same way for a pose database and for a benchmark
recording, so that the two can be compared as distributions.

A frame's feature is every joint's position and velocity in that frame's character frame,
plus the character frame's own ground velocity and turning rate. Everything is derived from
joint positions and rotations alone -- never from the velocities a database or a pipeline
stores -- so no method is judged by a quantity it computed itself.

Recordings come from a sweep run with ``recordFullPose`` on: the ``pose`` channel is the
pipeline's pose buffer copied verbatim each synthesis tick, laid out as its manifest describes.
That buffer keeps bone 0 in whatever space its source clip used, so the world pose is rebuilt the
way ``MotionSynthesisComponent`` renders it: bone 0 re-expressed in the pose's own character frame,
then placed under the character's recorded world frame (the ``root`` and ``rootForward`` channels).
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from core.pose_set import PoseSet
from core.simulation_frame import (clip_ranges, derive_frames, extend_by_one_frame,
                                   forward_kinematics, frame_rates, parent_indices,
                                   yaw_quaternion)

FULL_POSE_CHANNEL = 'pose'
TIME_CHANNEL = 'time'
FRAME_POSITION_CHANNEL = 'root'
FRAME_FORWARD_CHANNEL = 'rootForward'


class FeatureGroups(NamedTuple):
    """Column ranges of each feature group, which are normalised as a block."""
    positions: slice
    velocities: slice
    frame_rates: slice


def feature_groups(n_bones: int) -> FeatureGroups:
    return FeatureGroups(positions=slice(0, 3 * n_bones),
                         velocities=slice(3 * n_bones, 6 * n_bones),
                         frame_rates=slice(6 * n_bones, 6 * n_bones + 3))


def sequence_features(positions: np.ndarray,
                      rotations: np.ndarray,
                      parents: np.ndarray,
                      reference_bone_index: int,
                      forward_local: np.ndarray,
                      frame_time: float) -> np.ndarray:
    """
    Features of one contiguous pose sequence, one row per frame except the last.

    :param positions: (n, num_bones, 3) parent-local positions, slot 0 the root's world position.
    :param rotations: (n, num_bones, 4) parent-local rotations xyzw, slot 0 world.
    :return: (n - 1, 6 * num_bones + 3) float32. Row ``i`` describes frame ``i`` and the step
        out of it, so the sequence loses its last frame.
    """
    world_positions, _ = forward_kinematics(positions, rotations, parents)
    frames = derive_frames(positions, rotations, parents, reference_bone_index, forward_local)
    rates = frame_rates(frames, frame_time)

    inverse_frame = Rotation.from_quat(frames.rotation[:-1]).inv()
    n_frames, n_bones = world_positions.shape[0] - 1, world_positions.shape[1]

    relative = world_positions[:-1] - frames.position[:-1, np.newaxis, :]
    joint_velocity = (world_positions[1:] - world_positions[:-1]) / frame_time

    # One Rotation per (frame, bone) pair, so the frame's inverse is repeated across bones.
    per_bone_inverse = Rotation.from_quat(np.repeat(inverse_frame.as_quat(), n_bones, axis=0))
    local_positions = per_bone_inverse.apply(relative.reshape(-1, 3)).reshape(n_frames, n_bones * 3)
    local_velocities = per_bone_inverse.apply(joint_velocity.reshape(-1, 3)).reshape(n_frames, n_bones * 3)

    frame_motion = np.stack([rates.linear[:, 0], rates.linear[:, 2], rates.yaw], axis=1)

    return np.concatenate([local_positions, local_velocities, frame_motion], axis=1).astype(np.float32)


def database_sequences(pose_set: PoseSet) -> list[np.ndarray]:
    """Features of every clip in a database, one array per clip."""
    parents = parent_indices(pose_set.skeleton)
    frame_time = float(pose_set.frameTime)
    sequences = []
    for start, end in clip_ranges(pose_set.clips, pose_set.local_positions.shape[0]):
        # The extra frame lets every stored pose keep its feature row.
        positions, rotations = extend_by_one_frame(
            pose_set.local_positions[start:end], pose_set.local_rotations[start:end],
            pose_set.local_velocities[start:end], pose_set.local_angular_velocities[start:end],
            frame_time)
        sequences.append(sequence_features(positions, rotations, parents,
                                           int(pose_set.sim_frame_bone_index),
                                           pose_set.sim_frame_forward, frame_time))
    return sequences


class Recording(NamedTuple):
    name: str
    time: np.ndarray
    frame_time: float
    bone_names: list[str]
    positions: np.ndarray
    rotations: np.ndarray
    frame_position: np.ndarray
    frame_forward: np.ndarray


def read_recording(manifest_path: str) -> Recording:
    """
    Reads the time and full-pose channels of one benchmark recording.

    :raises ValueError: the recording has no full-pose channel, or stores rotations in a
        format other than quaternions.
    """
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)

    channels = {channel['name']: channel for channel in manifest['channels']}
    if FULL_POSE_CHANNEL not in channels:
        raise ValueError(f"{manifest_path} has no '{FULL_POSE_CHANNEL}' channel; rerun the sweep "
                         f"with recordFullPose enabled.")

    data_path = os.path.join(os.path.dirname(manifest_path), manifest['dataFile'])
    data = np.fromfile(data_path, dtype='<f4').reshape(manifest['frameCount'],
                                                       manifest['frameFloatCount'])

    pose_channel = channels[FULL_POSE_CHANNEL]
    layout = pose_channel['poseLayout']
    if layout['rotationStride'] != 4:
        raise ValueError(f"{manifest_path}: expected quaternion rotations, got "
                         f"{layout['rotationFormat']}.")

    n_bones = len(layout['boneNames'])
    pose = data[:, pose_channel['floatOffset']:pose_channel['floatOffset'] + pose_channel['floatCount']]
    positions = pose[:, layout['positionStart']:layout['positionStart'] + 3 * n_bones]
    rotations = pose[:, layout['rotationStart']:layout['rotationStart'] + 4 * n_bones]

    def channel(name: str) -> np.ndarray:
        offset = channels[name]['floatOffset']
        return data[:, offset:offset + channels[name]['floatCount']].copy()

    return Recording(name=manifest['recordingName'],
                     time=data[:, channels[TIME_CHANNEL]['floatOffset']].copy(),
                     frame_time=1.0 / float(manifest['synthesisFrameRate']),
                     bone_names=list(layout['boneNames']),
                     positions=positions.reshape(-1, n_bones, 3).copy(),
                     rotations=rotations.reshape(-1, n_bones, 4).copy(),
                     frame_position=channel(FRAME_POSITION_CHANNEL),
                     frame_forward=channel(FRAME_FORWARD_CHANNEL))


def recording_features(recording: Recording,
                       pose_set: PoseSet,
                       settle_time: float) -> np.ndarray:
    """
    Features of a recording after its settle period, measured on the database's skeleton.

    :raises ValueError: the recording's bones are not the database's, in the same order.
    """
    database_bones = [joint.name for joint in pose_set.skeleton]
    if recording.bone_names != database_bones:
        raise ValueError(f"{recording.name}: its bones do not match the database's skeleton.")

    parents = parent_indices(pose_set.skeleton)
    reference_bone_index = int(pose_set.sim_frame_bone_index)
    positions, rotations = world_pose(recording, parents, reference_bone_index,
                                      pose_set.sim_frame_forward)

    keep = recording.time >= settle_time
    return sequence_features(positions[keep], rotations[keep], parents, reference_bone_index,
                             pose_set.sim_frame_forward, recording.frame_time)


def world_pose(recording: Recording,
               parents: np.ndarray,
               reference_bone_index: int,
               forward_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The recorded poses with bone 0 moved from clip space to where it was rendered."""
    own_frames = derive_frames(recording.positions, recording.rotations, parents,
                               reference_bone_index, forward_local)
    inverse_own = Rotation.from_quat(own_frames.rotation).inv()
    root_local_position = inverse_own.apply(recording.positions[:, 0] - own_frames.position)
    root_local_rotation = inverse_own * Rotation.from_quat(recording.rotations[:, 0])

    world_yaw = np.arctan2(recording.frame_forward[:, 0], recording.frame_forward[:, 2])
    world_frame = Rotation.from_quat(yaw_quaternion(world_yaw))

    positions = recording.positions.astype(np.float64).copy()
    rotations = recording.rotations.astype(np.float64).copy()
    positions[:, 0] = recording.frame_position + world_frame.apply(root_local_position)
    rotations[:, 0] = (world_frame * root_local_rotation).as_quat()
    return positions, rotations


def windows(sequence: np.ndarray, length: int, stride: int) -> np.ndarray:
    """(m, length, features) windows cut from one sequence; empty when it is too short."""
    if sequence.shape[0] < length:
        return np.zeros((0, length, sequence.shape[1]), dtype=sequence.dtype)
    starts = np.arange(0, sequence.shape[0] - length + 1, stride)
    return np.stack([sequence[s:s + length] for s in starts])
