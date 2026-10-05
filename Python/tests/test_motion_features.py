"""Rebuilding a recorded pose's world placement, and the features measured from it."""

import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.motion_features import (Recording, sequence_features, windows,  # noqa: E402
                                       world_pose)
from core.simulation_frame import derive_frames, yaw_quaternion  # noqa: E402

PARENTS = np.array([-1, 0, 1])
FORWARD = np.array([0.0, 0.0, 1.0])
FRAME_TIME = 1.0 / 30.0


def walking_sequence(n=40):
    """A three-bone chain whose root walks a curve and bobs, in world space."""
    t = np.arange(n) * FRAME_TIME
    positions = np.zeros((n, 3, 3))
    positions[:, 0] = np.stack([np.sin(t), 0.9 + 0.02 * np.sin(8 * t), 2.0 * t], axis=1)
    positions[:, 1] = [0.0, 0.3, 0.0]
    positions[:, 2] = [0.0, 0.3, 0.1]
    rotations = np.zeros((n, 3, 4))
    rotations[:, 0] = (Rotation.from_quat(yaw_quaternion(0.5 * t)) *
                       Rotation.from_euler('x', 0.1 * np.sin(8 * t)[:, None])).as_quat()
    rotations[:, 1] = Rotation.from_euler('z', 0.3 * np.sin(4 * t)[:, None]).as_quat()
    rotations[:, 2] = [0.0, 0.0, 0.0, 1.0]
    return positions, rotations


class WorldPoseTests(unittest.TestCase):
    def test_restores_the_world_root_from_a_pose_left_in_clip_space(self):
        positions, rotations = walking_sequence()
        frames = derive_frames(positions, rotations, PARENTS, 0, FORWARD)

        # The pose buffer holds bone 0 wherever its source clip had it: here, every tick
        # displaced and turned by a different arbitrary amount.
        rng = np.random.default_rng(1)
        offset = Rotation.from_quat(yaw_quaternion(rng.uniform(-3, 3, len(positions))))
        shift = rng.uniform(-5, 5, (len(positions), 3)) * [1, 0, 1]
        clip_positions = positions.copy()
        clip_rotations = rotations.copy()
        clip_positions[:, 0] = offset.apply(positions[:, 0]) + shift
        clip_rotations[:, 0] = (offset * Rotation.from_quat(rotations[:, 0])).as_quat()

        forward = Rotation.from_quat(frames.rotation).apply(FORWARD)
        recording = Recording(name='test', time=np.arange(len(positions)) * FRAME_TIME,
                              frame_time=FRAME_TIME, bone_names=['a', 'b', 'c'],
                              positions=clip_positions, rotations=clip_rotations,
                              frame_position=frames.position, frame_forward=forward)

        rebuilt_positions, rebuilt_rotations = world_pose(recording, PARENTS, 0, FORWARD)

        np.testing.assert_allclose(rebuilt_positions, positions, atol=1e-5)
        same = np.abs(np.sum(rebuilt_rotations[:, 0] * rotations[:, 0], axis=1))
        np.testing.assert_allclose(same, 1.0, atol=1e-5)


class SequenceFeatureTests(unittest.TestCase):
    def test_features_do_not_depend_on_where_the_motion_happens(self):
        positions, rotations = walking_sequence()
        moved_positions, moved_rotations = positions.copy(), rotations.copy()
        turn = Rotation.from_euler('y', 1.2)
        moved_positions[:, 0] = turn.apply(positions[:, 0]) + [4.0, 0.0, -7.0]
        moved_rotations[:, 0] = (turn * Rotation.from_quat(rotations[:, 0])).as_quat()

        here = sequence_features(positions, rotations, PARENTS, 0, FORWARD, FRAME_TIME)
        there = sequence_features(moved_positions, moved_rotations, PARENTS, 0, FORWARD, FRAME_TIME)

        self.assertEqual(here.shape, (len(positions) - 1, 6 * 3 + 3))
        np.testing.assert_allclose(here, there, atol=1e-4)

    def test_frame_speed_is_the_root_ground_speed(self):
        positions, rotations = walking_sequence()
        features = sequence_features(positions, rotations, PARENTS, 0, FORWARD, FRAME_TIME)
        ground = np.diff(positions[:, 0, [0, 2]], axis=0) / FRAME_TIME
        np.testing.assert_allclose(np.linalg.norm(features[:, -3:-1], axis=1),
                                   np.linalg.norm(ground, axis=1), rtol=1e-4)


class WindowTests(unittest.TestCase):
    def test_cuts_strided_windows_and_nothing_from_a_short_sequence(self):
        sequence = np.arange(20, dtype=np.float32).reshape(10, 2)
        cut = windows(sequence, 4, 3)
        self.assertEqual(cut.shape, (3, 4, 2))
        np.testing.assert_array_equal(cut[1, 0], sequence[3])
        self.assertEqual(windows(sequence, 11, 1).shape, (0, 11, 2))


if __name__ == '__main__':
    unittest.main()
