"""
Packing a training set into the vectors a Learned Motion Matching model trains on.

The fixtures are the same rigid three-bone rig ``test_training_data`` uses, plus a synthetic
matching feature set, so every width and every offset is known exactly and a packing mistake shows
up as a number here rather than as bad motion much later.

The property worth stating: ``Y`` is joint-*local* and ``Q`` is character-space, and the two are
related by forward kinematics rather than being two independent predictions. Several tests below
check that relation directly, because it is what the paper's loss is built on and it is invisible
from either vector alone.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmm_dataset  # noqa: E402
import neural_packing  # noqa: E402
import training_data  # noqa: E402
from feature_set_importer import FeatureSchema, FeatureSet  # noqa: E402
from test_training_data import build_pose_set, walking_straight  # noqa: E402

# Two trajectory features of two floats over three horizons, then one three-float pose feature.
TRAJECTORY_FLOATS = 2 * (2 * 3)
POSE_FLOATS = 3
FEATURE_SIZE = TRAJECTORY_FLOATS + POSE_FLOATS


def build_feature_set(n_frames: int, invalid=()) -> FeatureSet:
    """A feature database of the right shape, with arbitrary but reproducible contents."""
    rng = np.random.default_rng(7)
    features = rng.standard_normal((n_frames, FEATURE_SIZE)).astype(np.float32)

    valid = np.ones(n_frames, dtype=bool)
    valid[list(invalid)] = False
    features[~valid] = 0.0

    schema = [
        FeatureSchema('futurePosition', 2, 3, True, 0),
        FeatureSchema('futureDirection', 2, 3, True, 6),
        FeatureSchema('leftFootVelocity', 3, 1, False, 12),
    ]
    return FeatureSet(features=features, valid=valid,
                      mean=np.zeros(FEATURE_SIZE, dtype=np.float32),
                      std=np.ones(FEATURE_SIZE, dtype=np.float32),
                      schema=schema)


def straight_walk_set(n_frames: int = 40, clips=None, invalid=(),
                      with_features: bool = True) -> training_data.TrainingSet:
    clips = clips or [(0, n_frames)]
    pose_set = build_pose_set(walking_straight(n_frames, speed=1.5),
                              np.zeros(n_frames, dtype=np.float32),
                              clips)
    feature_set = build_feature_set(n_frames, invalid) if with_features else None
    return training_data.build_training_set(pose_set, feature_set)


class LayoutTests(unittest.TestCase):
    def test_the_pose_vector_is_twelve_floats_a_bone_plus_the_roots_seven(self):
        layout = lmm_dataset.pose_vector_layout(5)

        self.assertEqual([name for name, _, _ in layout],
                         ['rotations_6d', 'velocities', 'angular_velocities',
                          'root_height', 'root_velocity', 'root_yaw_rate', 'contacts'])
        self.assertEqual(neural_packing.vector_size(layout), 12 * 5 + 7)

    def test_the_pose_vector_carries_no_joint_positions(self):
        # They are forward kinematics of what it does carry, so predicting them as well would let
        # the network describe a pose its own rotations do not reach.
        names = [name for name, _, _ in lmm_dataset.pose_vector_layout(5)]

        self.assertNotIn('positions', names)

    def test_the_character_vector_is_nine_floats_a_bone(self):
        layout = lmm_dataset.character_vector_layout(5)

        self.assertEqual([name for name, _, _ in layout], ['positions', 'rotations_6d'])
        self.assertEqual(neural_packing.vector_size(layout), 9 * 5)


class PoseVectorTests(unittest.TestCase):
    def setUp(self):
        self.training_set = straight_walk_set()
        self.spec = lmm_dataset.build_spec(self.training_set)
        self.y = lmm_dataset.pose_vector(self.training_set, self.spec)
        self.q = lmm_dataset.character_vector(self.training_set, self.spec)
        self.layout = self.spec.pose_layout()

    def test_the_roots_rotation_is_its_character_space_one(self):
        # The character frame is the root's parent, so joint-local and character-space coincide
        # there and nowhere else.
        local = neural_packing.block(self.y, self.layout, 'rotations_6d') \
            .reshape(-1, self.spec.n_bones, 6)
        character = neural_packing.block(
            self.q, self.spec.character_layout(), 'rotations_6d').reshape(-1, self.spec.n_bones, 6)

        np.testing.assert_allclose(local[:, 0], character[:, 0], rtol=0, atol=1e-5)

    def test_the_root_height_is_the_roots_own_y(self):
        np.testing.assert_allclose(
            neural_packing.block(self.y, self.layout, 'root_height').ravel(),
            self.training_set.positions[:, 0, 1], rtol=0, atol=1e-6)

    def test_the_root_blocks_come_through_untouched(self):
        np.testing.assert_allclose(
            neural_packing.block(self.y, self.layout, 'root_velocity'),
            self.training_set.root_velocity, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            neural_packing.block(self.y, self.layout, 'contacts'),
            self.training_set.contacts, rtol=0, atol=1e-6)

    def test_the_character_vector_is_the_training_sets_own_positions(self):
        np.testing.assert_allclose(
            neural_packing.block(self.q, self.spec.character_layout(), 'positions')
            .reshape(-1, self.spec.n_bones, 3),
            self.training_set.positions, rtol=0, atol=1e-6)

    def test_a_bone_subset_narrows_every_per_bone_block(self):
        spec = lmm_dataset.build_spec(self.training_set, [self.training_set.bone_names[-1]])
        y = lmm_dataset.pose_vector(self.training_set, spec)
        q = lmm_dataset.character_vector(self.training_set, spec)

        self.assertEqual(y.shape[1], 12 * spec.n_bones + 7)
        self.assertEqual(q.shape[1], 9 * spec.n_bones)


class SpecTests(unittest.TestCase):
    def test_it_reports_the_widths_the_networks_are_built_from(self):
        spec = lmm_dataset.build_spec(straight_walk_set(), latent_size=8)

        self.assertEqual(spec.feature_size, FEATURE_SIZE)
        self.assertEqual(spec.latent_size, 8)
        self.assertEqual(spec.pose_size, 12 * spec.n_bones + 7)
        self.assertEqual(spec.character_size, 9 * spec.n_bones)
        self.assertEqual(spec.compressor_input_size, spec.pose_size + spec.character_size)
        self.assertEqual(spec.decompressor_input_size, FEATURE_SIZE + 8)

    def test_it_carries_the_hierarchy_forward_kinematics_needs(self):
        spec = lmm_dataset.build_spec(straight_walk_set())

        self.assertEqual(int(spec.parents[0]), -1)
        self.assertEqual(spec.rest_offsets.shape, (spec.n_bones, 3))
        np.testing.assert_array_equal(spec.rest_offsets[0], 0.0)

    def test_it_refuses_a_database_with_no_matching_features(self):
        with self.assertRaises(ValueError) as raised:
            lmm_dataset.build_spec(straight_walk_set(with_features=False))
        self.assertIn('.mmfeatures', str(raised.exception))

    def test_excluding_a_bone_narrows_the_pose_vector(self):
        training_set = straight_walk_set()
        spec = lmm_dataset.build_spec(training_set, [training_set.bone_names[-1]])

        self.assertEqual(spec.n_bones, training_set.n_bones - 1)
        self.assertNotIn(training_set.bone_names[-1], spec.bone_names)


class CompressibleTests(unittest.TestCase):
    def test_the_last_frame_of_every_clip_has_no_latent(self):
        latent_exists = lmm_dataset.compressible(
            straight_walk_set(n_frames=40, clips=[(0, 25), (25, 40)]))

        self.assertFalse(latent_exists[24], 'the last frame of clip 0 has no successor to pair with')
        self.assertFalse(latent_exists[39])
        self.assertTrue(latent_exists[23])
        self.assertTrue(latent_exists[25])

    def test_an_invalid_feature_vector_removes_its_own_frame_and_the_one_before_it(self):
        latent_exists = lmm_dataset.compressible(straight_walk_set(n_frames=40, invalid=[10]))

        self.assertFalse(latent_exists[10], 'the frame itself is unusable')
        self.assertFalse(latent_exists[9], 'and it cannot be the successor of the frame before it')
        self.assertTrue(latent_exists[8])
        self.assertTrue(latent_exists[11])

    def test_a_clip_with_no_gait_cycle_is_kept(self):
        # The fixture authors no phase, so phase_rate is zero everywhere -- the sentinel
        # pfnn_dataset.usable_queries drops a frame on. A learned matcher wants idling kept.
        training_set = straight_walk_set()
        np.testing.assert_array_equal(training_set.phase_rate, 0.0)
        self.assertTrue(lmm_dataset.compressible(training_set)[0])


class TrainingPairTests(unittest.TestCase):
    def test_a_pair_needs_two_latents_so_it_loses_the_last_two_frames_of_a_clip(self):
        pairs = lmm_dataset.training_pairs(
            straight_walk_set(n_frames=40, clips=[(0, 25), (25, 40)]))

        self.assertNotIn(23, pairs)
        self.assertNotIn(24, pairs)
        self.assertIn(22, pairs)
        self.assertNotIn(38, pairs)
        self.assertIn(37, pairs)

    def test_every_pair_can_index_two_frames_past_itself(self):
        training_set = straight_walk_set(n_frames=40, clips=[(0, 25), (25, 40)])
        latent_exists = lmm_dataset.compressible(training_set)

        for frame in lmm_dataset.training_pairs(training_set):
            self.assertTrue(latent_exists[frame])
            self.assertTrue(latent_exists[frame + 1])


class StepperWindowTests(unittest.TestCase):
    """
    The runs the phase B stepper unrolls over.

    The property that matters is that a window never spans a cut: a stepper trained across one
    would learn to predict a jump, and jumping is the search's job. Nothing checks it at the point
    of use, so it is checked here.
    """

    def test_a_window_needs_a_latent_at_every_frame_it_covers(self):
        training_set = straight_walk_set(n_frames=40)
        latent_exists = lmm_dataset.compressible(training_set)

        for start in lmm_dataset.stepper_windows(training_set, window=5):
            for offset in range(6):
                self.assertTrue(latent_exists[start + offset],
                                f'window {start} reaches frame {start + offset}')

    def test_no_window_crosses_a_clip_boundary(self):
        training_set = straight_walk_set(n_frames=40, clips=[(0, 25), (25, 40)])

        for start in lmm_dataset.stepper_windows(training_set, window=5):
            self.assertEqual(start // 25, (start + 5) // 25,
                             f'window {start} spans the cut at frame 25')

    def test_a_longer_window_is_a_subset_of_a_shorter_one(self):
        training_set = straight_walk_set(n_frames=40, clips=[(0, 25), (25, 40)])

        short = set(lmm_dataset.stepper_windows(training_set, window=4).tolist())
        long = set(lmm_dataset.stepper_windows(training_set, window=8).tolist())

        self.assertTrue(long.issubset(short))
        self.assertLess(len(long), len(short))

    def test_a_window_longer_than_every_clip_leaves_nothing(self):
        training_set = straight_walk_set(n_frames=40, clips=[(0, 10), (10, 20), (20, 40)])

        self.assertEqual(lmm_dataset.stepper_windows(training_set, window=25).size, 0)

    def test_it_refuses_a_window_of_no_frames(self):
        with self.assertRaises(ValueError):
            lmm_dataset.latent_runs(np.ones(10, dtype=bool), 0)


class StateScaleTests(unittest.TestCase):
    def test_each_half_gets_one_scalar_over_its_latent_carrying_frames(self):
        x = np.tile(np.array([[1.0, 3.0]], dtype=np.float32), (6, 1))
        x[5] = 1000.0  # a frame with no latent, which must not reach the statistic
        latents = np.tile(np.array([[10.0, 30.0]], dtype=np.float32), (6, 1))
        exists = np.array([True] * 5 + [False])

        x_scale, z_scale = lmm_dataset.state_scales(x, latents, exists)

        self.assertAlmostEqual(x_scale, float(np.array([1.0, 3.0] * 5).std()), places=5)
        self.assertAlmostEqual(z_scale, float(np.array([10.0, 30.0] * 5).std()), places=4)

    def test_a_state_that_never_varies_gets_a_usable_scale_rather_than_zero(self):
        constant = np.ones((8, 3), dtype=np.float32)

        x_scale, z_scale = lmm_dataset.state_scales(constant, constant,
                                                    np.ones(8, dtype=bool))

        self.assertGreater(x_scale, 0.0)
        self.assertGreater(z_scale, 0.0)


class BuildVectorTests(unittest.TestCase):
    def setUp(self):
        self.training_set = straight_walk_set()
        self.spec = lmm_dataset.build_spec(self.training_set)

    def test_x_is_the_database_features_untouched(self):
        x, _, _, _ = lmm_dataset.build_vectors(self.training_set, self.spec)

        np.testing.assert_array_equal(x, self.training_set.features)

    def test_y_and_q_are_the_two_packings_over_the_specs_bones(self):
        _, y, q, _ = lmm_dataset.build_vectors(self.training_set, self.spec)

        np.testing.assert_allclose(y, lmm_dataset.pose_vector(self.training_set, self.spec),
                                   rtol=0, atol=1e-6)
        np.testing.assert_allclose(q, lmm_dataset.character_vector(self.training_set, self.spec),
                                   rtol=0, atol=1e-6)


class GroupNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.layout = neural_packing.layout([('wide', 3), ('narrow', 1)])
        rng = np.random.default_rng(3)
        self.vectors = (rng.standard_normal((200, 4)) * np.array([1.0, 5.0, 0.2, 3.0])) \
            .astype(np.float32)

    def test_the_mean_is_per_float(self):
        mean, _ = lmm_dataset.group_normalization(self.vectors, self.layout)

        np.testing.assert_allclose(mean, self.vectors.mean(axis=0), rtol=0, atol=1e-5)

    def test_one_standard_deviation_covers_a_whole_block(self):
        # Dividing a 3-vector by three different numbers warps it, so an L1 error on the result
        # means something different per axis. The paper scales a feature by one number.
        _, std = lmm_dataset.group_normalization(self.vectors, self.layout)

        self.assertEqual(len(set(std[:3].tolist())), 1)
        np.testing.assert_allclose(std[:3], self.vectors[:, :3].std(), rtol=1e-5, atol=0)
        np.testing.assert_allclose(std[3], self.vectors[:, 3].std(), rtol=1e-5, atol=0)

    def test_a_block_that_never_varies_is_left_alone_rather_than_divided_by_zero(self):
        constant = np.zeros((10, 4), dtype=np.float32)
        _, std = lmm_dataset.group_normalization(constant, self.layout)

        np.testing.assert_array_equal(std, 1.0)


class FeatureWeightTests(unittest.TestCase):
    def test_one_authored_weight_expands_across_every_float_of_its_feature(self):
        expanded = lmm_dataset.feature_weights(straight_walk_set(), [1.0, 2.0, 3.0])

        self.assertEqual(expanded.size, FEATURE_SIZE)
        np.testing.assert_array_equal(expanded[:6], 1.0)
        np.testing.assert_array_equal(expanded[6:12], 2.0)
        np.testing.assert_array_equal(expanded[12:], 3.0)

    def test_a_short_list_is_padded_with_ones_rather_than_silently_zeroing_a_feature(self):
        expanded = lmm_dataset.feature_weights(straight_walk_set(), [4.0])

        np.testing.assert_array_equal(expanded[:6], 4.0)
        np.testing.assert_array_equal(expanded[6:], 1.0)

    def test_it_refuses_a_database_with_no_feature_schema(self):
        with self.assertRaises(ValueError):
            lmm_dataset.feature_weights(straight_walk_set(with_features=False), [1.0])


if __name__ == '__main__':
    unittest.main()
