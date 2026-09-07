"""
Packing a training set into PFNN input and output vectors.

The fixtures are the same rigid three-bone rig ``test_training_data`` uses, for the same reason: a
character carried along by its own frame has no motion inside it, so the right answer is known
exactly and a packing mistake shows up as a number rather than as bad motion much later.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pfnn_dataset  # noqa: E402
import training_data  # noqa: E402
from test_training_data import (FRAME_TIME, build_pose_set,  # noqa: E402
                                walking_straight)

# Frames per half-cycle, so a fixture walks with alternating footfalls and a phase that turns over
# steadily. A clip with no measurable cycle carries a rate of zero, and pfnn_dataset drops those
# frames on purpose.
CONTACT_PERIOD = 10


def contacts_for(n_frames: int) -> np.ndarray:
    """Alternating footfalls, so every frame belongs to a measurable gait cycle."""
    contacts = np.zeros((n_frames, 2), dtype=bool)
    for frame in range(n_frames):
        contacts[frame, (frame // CONTACT_PERIOD) % 2] = \
            frame % CONTACT_PERIOD < CONTACT_PERIOD // 2
    return contacts


def phase_for(n_frames: int):
    """
    The steady cycle those footfalls imply -- half a turn per footfall -- as Unity would have
    evaluated and written it. Authored here because this side no longer derives phase.
    """
    slope = np.pi / CONTACT_PERIOD
    unwrapped = np.arange(n_frames, dtype=np.float64) * slope
    return (np.mod(unwrapped, 2.0 * np.pi).astype(np.float32),
            np.full(n_frames, slope / FRAME_TIME, dtype=np.float32))


def straight_walk_set(n_frames: int = 80, speed: float = 1.5,
                      clips=None) -> training_data.TrainingSet:
    clips = clips or [(0, n_frames)]
    phase, phase_rate = phase_for(n_frames)
    pose_set = build_pose_set(walking_straight(n_frames, speed),
                              np.zeros(n_frames, dtype=np.float32),
                              clips,
                              foot_contacts=contacts_for(n_frames),
                              phase=phase,
                              phase_rate=phase_rate)
    return training_data.build_training_set(pose_set)


class BoneSelectionTests(unittest.TestCase):
    NAMES = ['root', 'spine', 'foot']
    PARENTS = np.array([-1, 0, 1])

    def test_an_empty_exclusion_keeps_every_bone(self):
        selected = pfnn_dataset.select_bones(self.NAMES, self.PARENTS, [])
        np.testing.assert_array_equal(selected, [0, 1, 2])

    def test_selection_is_in_skeleton_order_not_exclusion_order(self):
        selected = pfnn_dataset.select_bones(['root', 'a', 'b', 'c'],
                                             np.array([-1, 0, 0, 0]), ['c', 'a'])
        np.testing.assert_array_equal(selected, [0, 2])

    def test_an_unknown_exclusion_is_an_error_not_a_no_op(self):
        # Silently ignoring it would train a different model than the config describes, and the
        # first symptom would be a checkpoint that no longer loads.
        with self.assertRaises(ValueError) as raised:
            pfnn_dataset.select_bones(self.NAMES, self.PARENTS, ['sipne'])
        self.assertIn('sipne', str(raised.exception))

    def test_the_root_cannot_be_excluded(self):
        with self.assertRaises(ValueError):
            pfnn_dataset.select_bones(self.NAMES, self.PARENTS, ['root'])

    def test_keeping_a_bone_whose_parent_is_excluded_is_refused(self):
        # A rotation needs a frame to be applied in, and that frame is the parent's.
        with self.assertRaises(ValueError) as raised:
            pfnn_dataset.select_bones(self.NAMES, self.PARENTS, ['spine'])
        self.assertIn('foot', str(raised.exception))


class WindowOffsetTests(unittest.TestCase):
    def test_the_default_window_is_symmetric_and_includes_the_present(self):
        offsets = pfnn_dataset.default_window_offsets(30, 5)
        np.testing.assert_array_equal(offsets, [-30, -25, -20, -15, -10, -5,
                                                0, 5, 10, 15, 20, 25, 30])

    def test_a_stride_that_does_not_divide_the_radius_still_includes_zero(self):
        offsets = pfnn_dataset.default_window_offsets(10, 4)
        np.testing.assert_array_equal(offsets, [-8, -4, 0, 4, 8])


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.spec = pfnn_dataset.build_spec(straight_walk_set(), excluded_bones=[])

    def test_blocks_tile_the_vector_without_gaps_or_overlap(self):
        for layout, size in ((self.spec.input_layout(), self.spec.input_size),
                             (self.spec.output_layout(), self.spec.output_size)):
            expected = 0
            for _, offset, count in layout:
                self.assertEqual(offset, expected)
                expected += count
            self.assertEqual(expected, size)

    def test_the_declared_sizes_match_what_build_vectors_packs(self):
        x, y, _, _ = pfnn_dataset.build_vectors(straight_walk_set(), self.spec)
        self.assertEqual(x.shape[1], self.spec.input_size)
        self.assertEqual(y.shape[1], self.spec.output_size)

    def test_naming_a_block_that_does_not_exist_is_an_error(self):
        with self.assertRaises(KeyError):
            pfnn_dataset.block(np.zeros(self.spec.input_size),
                               self.spec.input_layout(), 'terrain_heights')


class SampleSelectionTests(unittest.TestCase):
    def test_no_sample_takes_its_target_from_the_next_clip(self):
        n_frames = 80
        clips = [(0, 40), (40, n_frames)]
        training_set = straight_walk_set(n_frames, clips=clips)

        _, _, _, frames = pfnn_dataset.build_vectors(
            training_set, pfnn_dataset.build_spec(training_set))

        # Frame 39 is the last of its clip: predicting frame 40 would report a jump cut as motion.
        self.assertNotIn(39, frames.tolist())
        self.assertNotIn(n_frames - 1, frames.tolist())

    def test_frames_with_no_measurable_gait_cycle_are_dropped(self):
        # A clip with no measurable cycle is marked by a phase rate of zero, and
        # TrainingSet.usable() does not check that -- so the packing has to.
        n_frames = 40
        pose_set = build_pose_set(walking_straight(n_frames, 1.5),
                                  np.zeros(n_frames, dtype=np.float32),
                                  [(0, n_frames)],
                                  foot_contacts=np.zeros((n_frames, 2), dtype=bool))
        training_set = training_data.build_training_set(pose_set)

        self.assertTrue(np.all(training_set.phase_rate == 0.0))
        self.assertFalse(np.any(pfnn_dataset.usable_queries(training_set)))


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.speed = 1.5
        self.training_set = straight_walk_set(80, self.speed)
        self.spec = pfnn_dataset.build_spec(self.training_set)
        self.x, self.y, self.phase, self.frames = pfnn_dataset.build_vectors(
            self.training_set, self.spec)

    def test_the_root_delta_is_one_frame_of_the_root_velocity(self):
        delta = pfnn_dataset.block(self.y, self.spec.output_layout(), 'root_delta')
        step = np.hypot(delta[:, 0], delta[:, 1])
        expected = np.linalg.norm(
            self.training_set.root_velocity[self.frames][:, [0, 2]], axis=1) * FRAME_TIME
        np.testing.assert_allclose(step, expected, atol=1e-6)

    def test_a_straight_walk_turns_by_nothing(self):
        delta = pfnn_dataset.block(self.y, self.spec.output_layout(), 'root_delta')
        np.testing.assert_allclose(delta[:, 2], 0.0, atol=1e-6)

    def test_the_phase_delta_is_one_frame_of_the_phase_rate(self):
        delta = pfnn_dataset.block(self.y, self.spec.output_layout(), 'phase_delta')[:, 0]
        expected = self.training_set.phase_rate[self.frames] * FRAME_TIME
        np.testing.assert_allclose(delta, expected, atol=1e-6)

    def test_the_present_sample_of_the_window_is_the_origin(self):
        # The window is expressed in the query frame's own space, so its offset-0 sample can only be
        # the origin facing +z. A non-zero value here means the frame transform was skipped.
        layout = self.spec.input_layout()
        present = list(self.spec.window_offsets).index(0)

        positions = pfnn_dataset.block(self.x, layout, 'trajectory_positions') \
            .reshape(-1, self.spec.n_offsets, 2)
        directions = pfnn_dataset.block(self.x, layout, 'trajectory_directions') \
            .reshape(-1, self.spec.n_offsets, 2)

        np.testing.assert_allclose(positions[:, present], 0.0, atol=1e-6)
        np.testing.assert_allclose(directions[:, present, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(directions[:, present, 1], 1.0, atol=1e-6)

    def test_the_rotation_block_round_trips_back_to_the_stored_rotations(self):
        six = pfnn_dataset.block(self.y, self.spec.output_layout(), 'joint_rotations_6d') \
            .reshape(-1, self.spec.n_bones, 6)
        recovered = training_data.rotations_from_6d(six)
        expected = self.training_set.rotations[self.frames + 1][:, self.spec.bone_indices]
        np.testing.assert_allclose(recovered, expected, atol=1e-5)


class NormalizationTests(unittest.TestCase):
    def test_a_constant_column_is_left_alone_rather_than_divided_by_zero(self):
        vectors = np.stack([np.array([1.0, 5.0]), np.array([3.0, 5.0])]).astype(np.float32)
        mean, std = pfnn_dataset.normalization(vectors)

        np.testing.assert_allclose(mean, [2.0, 5.0])
        self.assertEqual(std[1], 1.0)
        self.assertTrue(np.all(np.isfinite((vectors - mean) / std)))


if __name__ == '__main__':
    unittest.main()
