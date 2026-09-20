"""
The ``.lmm.npz`` checkpoint format.

Two properties matter more than the round trip: that ``stages_trained`` describes what the file
actually carries, and that retraining the autoencoder drops the stepper and projector fitted
against the latent space it just replaced. Both are enforced in the writer, so both are tested
against the writer rather than against a caller remembering to do the right thing.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmm_io  # noqa: E402

FEATURE_SIZE = 15
LATENT_SIZE = 4
POSE_SIZE = 21
CHARACTER_SIZE = 18
N_FRAMES = 12


def layers(sizes):
    """One weight/bias pair per consecutive size pair, filled distinctly so a swap is visible."""
    weights = [np.full((out, inp), i + 1, dtype=np.float32)
               for i, (inp, out) in enumerate(zip(sizes, sizes[1:]))]
    biases = [np.full(out, -(i + 1), dtype=np.float32)
              for i, out in enumerate(sizes[1:])]
    return weights, biases


def autoencoder_arguments() -> dict:
    compressor_weights, compressor_biases = layers([POSE_SIZE + CHARACTER_SIZE, 8, LATENT_SIZE])
    decompressor_weights, decompressor_biases = layers([FEATURE_SIZE + LATENT_SIZE, 8, POSE_SIZE])

    return dict(
        compressor_weights=compressor_weights, compressor_biases=compressor_biases,
        decompressor_weights=decompressor_weights, decompressor_biases=decompressor_biases,
        x_mean=np.arange(FEATURE_SIZE, dtype=np.float32),
        x_std=np.ones(FEATURE_SIZE, dtype=np.float32),
        y_mean=np.arange(POSE_SIZE, dtype=np.float32),
        y_std=np.ones(POSE_SIZE, dtype=np.float32),
        q_mean=np.arange(CHARACTER_SIZE, dtype=np.float32),
        q_std=np.ones(CHARACTER_SIZE, dtype=np.float32),
        z_mean=np.zeros(LATENT_SIZE, dtype=np.float32),
        z_std=np.ones(LATENT_SIZE, dtype=np.float32),
        latents=np.arange(N_FRAMES * LATENT_SIZE, dtype=np.float32).reshape(N_FRAMES, LATENT_SIZE),
        latent_valid=np.array([True] * (N_FRAMES - 1) + [False]),
        feature_weights=np.full(FEATURE_SIZE, 2.0, dtype=np.float32),
        bone_names=['Hips', 'Spine'],
        output_blocks=['rotations_6d', 'root_height'],
        character_blocks=['positions', 'rotations_6d'],
        parents=np.array([-1, 0], dtype=np.int32),
        rest_offsets=np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
        feature_names=['futurePosition', 'leftFootVelocity'],
        feature_widths=np.array([2, 3], dtype=np.int32),
        feature_counts=np.array([6, 1], dtype=np.int32),
        n_trajectory_features=1,
        pose_offset=12,
        latent_size=LATENT_SIZE,
        frame_time=1.0 / 60.0,
        n_frames=N_FRAMES,
        losses=[(0.5, 0.1, 0.2, 0.8, 0.9), (0.4, 0.1, 0.1, 0.6, 0.7)],
        loss_columns=('local', 'character', 'local_velocity', 'total', 'validation'))


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, 'test.lmm.npz')
        self.arguments = autoencoder_arguments()
        lmm_io.save_checkpoint(self.path, **self.arguments)
        self.loaded = lmm_io.load_checkpoint(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_it_loads(self):
        self.assertIsNotNone(self.loaded)

    def test_every_network_layer_comes_back_in_order(self):
        for stored, expected in zip(self.loaded.compressor_weights,
                                    self.arguments['compressor_weights']):
            np.testing.assert_array_equal(stored, expected)
        for stored, expected in zip(self.loaded.decompressor_biases,
                                    self.arguments['decompressor_biases']):
            np.testing.assert_array_equal(stored, expected)

    def test_the_latent_table_and_its_validity_survive(self):
        np.testing.assert_array_equal(self.loaded.latents, self.arguments['latents'])
        np.testing.assert_array_equal(self.loaded.latent_valid, self.arguments['latent_valid'])

    def test_the_packing_description_survives(self):
        self.assertEqual(self.loaded.bone_names, ['Hips', 'Spine'])
        self.assertEqual(self.loaded.output_blocks, ['rotations_6d', 'root_height'])
        self.assertEqual(self.loaded.character_blocks, ['positions', 'rotations_6d'])
        self.assertEqual(self.loaded.feature_names, ['futurePosition', 'leftFootVelocity'])
        self.assertEqual(self.loaded.n_trajectory_features, 1)
        self.assertEqual(self.loaded.pose_offset, 12)
        self.assertEqual(self.loaded.n_frames, N_FRAMES)
        self.assertAlmostEqual(self.loaded.frame_time, 1.0 / 60.0, places=6)

    def test_the_widths_are_derived_from_the_statistics_rather_than_stored_twice(self):
        self.assertEqual(self.loaded.feature_size, FEATURE_SIZE)
        self.assertEqual(self.loaded.pose_size, POSE_SIZE)
        self.assertEqual(self.loaded.latent_size, LATENT_SIZE)

    def test_the_loss_history_keeps_its_column_names(self):
        self.assertEqual(self.loaded.losses.shape, (2, 5))
        self.assertEqual(self.loaded.loss_columns[-1], 'validation')
        self.assertAlmostEqual(self.loaded.final_loss, 0.7, places=5)

    def test_it_loads_without_allow_pickle(self):
        # Reading a checkpoint must never mean executing what is inside it, so every string array
        # has to be fixed-width unicode rather than an object array.
        with np.load(self.path, allow_pickle=False) as data:
            self.assertEqual(list(data['bone_names']), ['Hips', 'Spine'])


class StagesTrainedTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, 'test.lmm.npz')

    def tearDown(self):
        self.directory.cleanup()

    def _save(self, **extra):
        lmm_io.save_checkpoint(self.path, **autoencoder_arguments(), **extra)
        return lmm_io.load_checkpoint(self.path)

    def test_the_autoencoder_alone_claims_only_itself(self):
        loaded = self._save()

        self.assertEqual(loaded.stages_trained, [lmm_io.STAGE_AUTOENCODER])
        self.assertTrue(loaded.has_stage(lmm_io.STAGE_AUTOENCODER))
        self.assertFalse(loaded.has_stage(lmm_io.STAGE_STEPPER))

    def test_a_stepper_is_claimed_only_when_its_parameters_are_there(self):
        weights, biases = layers([FEATURE_SIZE + LATENT_SIZE, 8, FEATURE_SIZE + LATENT_SIZE])
        loaded = self._save(stepper_weights=weights, stepper_biases=biases,
                            xz_rate_mean=np.zeros(FEATURE_SIZE + LATENT_SIZE, dtype=np.float32),
                            xz_rate_std=np.ones(FEATURE_SIZE + LATENT_SIZE, dtype=np.float32))

        self.assertEqual(loaded.stages_trained,
                         [lmm_io.STAGE_AUTOENCODER, lmm_io.STAGE_STEPPER])
        self.assertEqual(len(loaded.stepper_weights), 2)
        self.assertIsNotNone(loaded.xz_rate_std)

    def test_a_projector_without_a_stepper_is_refused(self):
        weights, biases = layers([FEATURE_SIZE, 8, FEATURE_SIZE + LATENT_SIZE])

        with self.assertRaises(ValueError):
            self._save(projector_weights=weights, projector_biases=biases)

    def test_retraining_the_autoencoder_drops_the_later_stages(self):
        weights, biases = layers([FEATURE_SIZE + LATENT_SIZE, 8, FEATURE_SIZE + LATENT_SIZE])
        self._save(stepper_weights=weights, stepper_biases=biases)

        # A second write with the autoencoder alone: the stepper was fitted against a latent space
        # that no longer exists, so keeping it would leave a file whose parts describe two models.
        retrained = self._save()

        self.assertEqual(retrained.stages_trained, [lmm_io.STAGE_AUTOENCODER])
        self.assertEqual(retrained.stepper_weights, [])
        self.assertIsNone(retrained.xz_rate_mean)


class LoadFailureTests(unittest.TestCase):
    def test_a_missing_file_is_none_rather_than_an_exception(self):
        self.assertIsNone(lmm_io.load_checkpoint('no/such/checkpoint.lmm.npz', log=lambda _: None))

    def test_an_empty_path_is_none(self):
        self.assertIsNone(lmm_io.load_checkpoint('', log=lambda _: None))

    def test_a_file_missing_a_key_is_reported_and_refused(self):
        messages = []
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'stale.lmm.npz')
            np.savez(path, x_mean=np.zeros(3, dtype=np.float32))

            self.assertIsNone(lmm_io.load_checkpoint(path, log=messages.append))

        self.assertEqual(len(messages), 1)
        self.assertIn('could not read checkpoint', messages[0])


if __name__ == '__main__':
    unittest.main()
