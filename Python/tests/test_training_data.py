"""
Building the per-frame arrays a neural motion model trains on.

The fixtures are rigid translations and rotations of a three-bone rig, because those are
the cases where the right answer is known exactly: a character carried along by its own
frame has no motion *within* that frame, whatever the world says it is doing.
"""

import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import training_data  # noqa: E402
from Animation import PoseSet  # noqa: E402
from Skeleton import Joint, Skeleton  # noqa: E402

FRAME_TIME = 1.0 / 30.0
ROOT_HEIGHT = 0.9


def build_skeleton() -> Skeleton:
    """root -> spine -> foot, a metre of spine above the root and a foot below it."""
    root = Joint('root', np.zeros(3, dtype=np.float32), Rotation.identity())
    spine = Joint('spine', np.array([0.0, 1.0, 0.0], dtype=np.float32), Rotation.identity())
    foot = Joint('foot', np.array([0.1, -0.9, 0.0], dtype=np.float32), Rotation.identity())
    root.add_child(spine)
    spine.add_child(foot)
    return Skeleton('test', root)


def build_pose_set(root_positions: np.ndarray,
                   root_yaws: np.ndarray,
                   clips,
                   foot_contacts: np.ndarray | None = None) -> PoseSet:
    """
    A database whose rig is rigid: every bone holds its rest offset and identity rotation,
    and only bone 0 moves. Stored velocities are the per-channel differences the extractor
    would have written, including the last frame of each clip, which is what
    ``extend_by_one_frame`` reconstructs the frame past the end from.
    """
    skeleton = build_skeleton()
    joints = list(skeleton)
    n_frames, n_bones = root_positions.shape[0], len(joints)

    positions = np.zeros((n_frames, n_bones, 3), dtype=np.float32)
    rotations = np.zeros((n_frames, n_bones, 4), dtype=np.float32)
    positions[:, 0] = root_positions
    rotations[:, 0] = Rotation.from_euler('y', root_yaws[:, np.newaxis]).as_quat()
    for bone in range(1, n_bones):
        positions[:, bone] = joints[bone].default_local_position
        rotations[:, bone] = Rotation.identity().as_quat()

    velocities = np.zeros_like(positions)
    angular_velocities = np.zeros((n_frames, n_bones, 3), dtype=np.float32)
    for start, end in clips:
        velocities[start:end - 1] = (positions[start + 1:end] - positions[start:end - 1]) / FRAME_TIME
        velocities[end - 1] = velocities[max(start, end - 2)]

        delta = (Rotation.from_quat(rotations[start + 1:end].reshape(-1, 4)) *
                 Rotation.from_quat(rotations[start:end - 1].reshape(-1, 4)).inv())
        angular_velocities[start:end - 1] = delta.as_rotvec().reshape(-1, n_bones, 3) / FRAME_TIME
        angular_velocities[end - 1] = angular_velocities[max(start, end - 2)]

    if foot_contacts is None:
        foot_contacts = np.zeros((n_frames, 2), dtype=bool)

    pose_set = PoseSet(skeleton=skeleton,
                       frame_time=FRAME_TIME,
                       local_pos=positions,
                       local_quats=rotations,
                       foot_contacts=foot_contacts,
                       local_vel=velocities,
                       local_angular_vel=angular_velocities,
                       clips=[{'start': start, 'end': end} for start, end in clips])
    pose_set.sim_frame_bone_index = 0
    pose_set.sim_frame_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return pose_set


def walking_straight(n_frames: int, speed: float, start_x: float = 0.0) -> np.ndarray:
    """Root positions marching along +x at a constant speed, at a fixed height."""
    x = start_x + np.arange(n_frames, dtype=np.float32) * speed * FRAME_TIME
    return np.stack([x, np.full(n_frames, ROOT_HEIGHT, dtype=np.float32), np.zeros(n_frames)],
                    axis=1).astype(np.float32)


class CharacterSpaceTests(unittest.TestCase):
    def setUp(self):
        self.speed = 1.5
        pose_set = build_pose_set(walking_straight(40, self.speed),
                                  np.zeros(40, dtype=np.float32), [(0, 40)])
        self.training_set = training_data.build_training_set(pose_set)

    def test_the_root_bone_sits_over_the_frame_origin(self):
        # The character frame is bone 0 ground-projected, so bone 0 can only be directly
        # above it -- if this drifts, the frame transform is wrong.
        root = self.training_set.positions[:, 0]

        np.testing.assert_allclose(root[:, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(root[:, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(root[:, 1], ROOT_HEIGHT, atol=1e-5)

    def test_a_rigid_translation_leaves_no_motion_inside_the_frame(self):
        np.testing.assert_allclose(self.training_set.velocities, 0.0, atol=1e-4)
        np.testing.assert_allclose(self.training_set.angular_velocities, 0.0, atol=1e-4)

    def test_the_frame_carries_the_travel_instead(self):
        # Identity facing means the frame is the world, so +x travel stays +x.
        np.testing.assert_allclose(self.training_set.root_velocity[:, 0], self.speed, atol=1e-3)
        np.testing.assert_allclose(self.training_set.root_velocity[:, [1, 2]], 0.0, atol=1e-3)
        np.testing.assert_allclose(self.training_set.root_yaw_rate, 0.0, atol=1e-4)

    def test_bones_keep_their_rest_offsets(self):
        spine = self.training_set.positions[:, 1]

        np.testing.assert_allclose(spine[:, 1], ROOT_HEIGHT + 1.0, atol=1e-5)


class TurningTests(unittest.TestCase):
    def test_a_turn_in_place_shows_up_only_as_a_yaw_rate(self):
        n_frames = 40
        rate = 0.5  # rad/s
        yaws = np.arange(n_frames, dtype=np.float32) * rate * FRAME_TIME
        positions = np.stack([np.zeros(n_frames), np.full(n_frames, ROOT_HEIGHT),
                              np.zeros(n_frames)], axis=1).astype(np.float32)

        training_set = training_data.build_training_set(
            build_pose_set(positions, yaws, [(0, n_frames)]))

        np.testing.assert_allclose(training_set.root_yaw_rate, rate, atol=1e-3)
        np.testing.assert_allclose(training_set.root_velocity, 0.0, atol=1e-3)
        np.testing.assert_allclose(training_set.velocities, 0.0, atol=1e-3)


class ClipBoundaryTests(unittest.TestCase):
    def test_no_rate_is_taken_across_a_clip_boundary(self):
        speed = 1.5
        first = walking_straight(20, speed, start_x=0.0)
        second = walking_straight(20, speed, start_x=100.0)
        positions = np.concatenate([first, second])

        training_set = training_data.build_training_set(
            build_pose_set(positions, np.zeros(40, dtype=np.float32), [(0, 20), (20, 40)]))

        # Frame 19 is the last of clip 0; frame 20 is 100 metres away in another animation.
        np.testing.assert_allclose(training_set.root_velocity[19, 0], speed, atol=1e-3)
        np.testing.assert_allclose(training_set.velocities[19], 0.0, atol=1e-3)


class PoseVectorTests(unittest.TestCase):
    def setUp(self):
        pose_set = build_pose_set(walking_straight(12, 1.0), np.zeros(12, dtype=np.float32),
                                  [(0, 12)])
        self.training_set = training_data.build_training_set(pose_set)

    def test_the_layout_covers_the_vector_exactly(self):
        vector = self.training_set.pose_vector()
        layout = self.training_set.pose_vector_layout()

        self.assertEqual(vector.shape, (12, 15 * self.training_set.n_bones + 6))
        self.assertEqual(layout[0][1], 0)
        self.assertEqual(layout[-1][1] + layout[-1][2], vector.shape[1])

    def test_blocks_line_up_with_the_arrays_they_came_from(self):
        vector = self.training_set.pose_vector()
        blocks = dict((name, (offset, count)) for name, offset, count in
                      self.training_set.pose_vector_layout())

        offset, count = blocks['positions']
        np.testing.assert_allclose(vector[:, offset:offset + count],
                                   self.training_set.positions.reshape(12, -1), atol=1e-6)

        offset, count = blocks['contacts']
        np.testing.assert_allclose(vector[:, offset:offset + count],
                                   self.training_set.contacts, atol=1e-6)


class RotationRepresentationTests(unittest.TestCase):
    def test_the_two_axis_form_reconstructs_the_rotation(self):
        rotations = Rotation.random(16, rng=np.random.default_rng(7)).as_quat()

        six = training_data.rotations_to_6d(rotations)

        first, second = six[:, :3], six[:, 3:]
        third = np.cross(first, second)
        matrices = np.stack([first, second, third], axis=2)
        recovered = Rotation.from_matrix(matrices).as_quat()

        # Up to sign: q and -q are the same rotation.
        self.assertTrue(np.allclose(recovered, rotations, atol=1e-5) or
                        np.allclose(recovered, -rotations, atol=1e-5) or
                        np.allclose(np.abs((recovered * rotations).sum(axis=1)), 1.0, atol=1e-5))


class TrajectoryWindowTests(unittest.TestCase):
    """
    A trajectory window is the one thing a model needs that the frame transform removes, so
    these check it is measured in the query frame and never leaves the query frame's clip.
    """

    OFFSETS = [-10, -5, 0, 5, 10]

    def test_a_straight_walk_lays_the_window_out_along_the_travel_axis(self):
        speed = 1.5
        training_set = training_data.build_training_set(
            build_pose_set(walking_straight(60, speed), np.zeros(60, dtype=np.float32), [(0, 60)]))

        positions, directions = training_set.trajectory_window(self.OFFSETS)

        # Facing is identity (+z) while travel is +x, so the window runs along local x.
        step = speed * FRAME_TIME
        np.testing.assert_allclose(positions[30, :, 0], np.array(self.OFFSETS) * step, atol=1e-4)
        np.testing.assert_allclose(positions[30, :, 1], 0.0, atol=1e-4)
        # Nothing turns, so every sample faces the way the query frame does.
        np.testing.assert_allclose(directions[30], np.tile([0.0, 1.0], (len(self.OFFSETS), 1)),
                                   atol=1e-4)

    def test_the_window_is_measured_in_the_query_frames_own_space(self):
        # The same walk started somewhere else and facing somewhere else must produce the
        # same window -- that invariance is the whole point of measuring it in the frame.
        speed = 1.5
        straight = training_data.build_training_set(
            build_pose_set(walking_straight(60, speed), np.zeros(60, dtype=np.float32), [(0, 60)]))

        yaw = 0.9
        positions = walking_straight(60, speed, start_x=17.0)
        rotated = positions.copy()
        rotated[:, 0] = np.cos(yaw) * positions[:, 0] + np.sin(yaw) * positions[:, 2]
        rotated[:, 2] = -np.sin(yaw) * positions[:, 0] + np.cos(yaw) * positions[:, 2]
        turned = training_data.build_training_set(
            build_pose_set(rotated, np.full(60, yaw, dtype=np.float32), [(0, 60)]))

        np.testing.assert_allclose(turned.trajectory_window(self.OFFSETS)[0][30],
                                   straight.trajectory_window(self.OFFSETS)[0][30], atol=1e-3)

    def test_a_turn_shows_up_as_the_window_facing_turning(self):
        n_frames = 60
        rate = 0.5
        yaws = np.arange(n_frames, dtype=np.float32) * rate * FRAME_TIME
        positions = np.stack([np.zeros(n_frames), np.full(n_frames, ROOT_HEIGHT),
                              np.zeros(n_frames)], axis=1).astype(np.float32)
        training_set = training_data.build_training_set(
            build_pose_set(positions, yaws, [(0, n_frames)]))

        _, directions = training_set.trajectory_window([0, 10])

        expected = rate * 10 * FRAME_TIME
        self.assertAlmostEqual(float(np.arctan2(directions[30, 1, 0], directions[30, 1, 1])),
                               expected, places=3)

    def test_offsets_are_clamped_inside_the_query_frames_own_clip(self):
        speed = 1.5
        first = walking_straight(30, speed, start_x=0.0)
        second = walking_straight(30, speed, start_x=100.0)
        training_set = training_data.build_training_set(
            build_pose_set(np.concatenate([first, second]),
                           np.zeros(60, dtype=np.float32), [(0, 30), (30, 60)]))

        positions, _ = training_set.trajectory_window([-10, 0, 10])

        # Frame 29 is the last of clip 0; frame 30 is 100 metres away in another animation.
        self.assertLess(abs(positions[29, 2, 0]), 1.0)
        # Frame 0 has no past inside its clip, so the window repeats it.
        np.testing.assert_allclose(positions[0, 0], positions[0, 1], atol=1e-5)

    def test_the_window_is_shaped_by_the_offsets_it_was_asked_for(self):
        training_set = training_data.build_training_set(
            build_pose_set(walking_straight(20, 1.0), np.zeros(20, dtype=np.float32), [(0, 20)]))

        positions, directions = training_set.trajectory_window([-4, 0, 4])

        self.assertEqual(positions.shape, (20, 3, 2))
        self.assertEqual(directions.shape, (20, 3, 2))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=2), 1.0, atol=1e-5)


class RoundTripTests(unittest.TestCase):
    def test_an_npz_reads_back_as_the_same_training_set(self):
        import tempfile

        contacts = np.zeros((30, 2), dtype=bool)
        contacts[5:12, 0] = True
        contacts[15:22, 1] = True
        pose_set = build_pose_set(walking_straight(30, 1.2), np.zeros(30, dtype=np.float32),
                                  [(0, 30)], foot_contacts=contacts)
        original = training_data.build_training_set(pose_set)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'training.npz')
            training_data.save_npz(original, path)
            restored = training_data.load_npz(path)

        self.assertEqual(restored.bone_names, original.bone_names)
        self.assertAlmostEqual(restored.frame_time, original.frame_time, places=6)
        np.testing.assert_array_equal(restored.clip_ranges, original.clip_ranges)
        np.testing.assert_allclose(restored.positions, original.positions)
        np.testing.assert_allclose(restored.rotations, original.rotations)
        np.testing.assert_allclose(restored.phase, original.phase)
        np.testing.assert_allclose(restored.frame_position, original.frame_position)
        np.testing.assert_allclose(restored.frame_yaw, original.frame_yaw)
        self.assertIsNone(restored.features)


if __name__ == '__main__':
    unittest.main()
