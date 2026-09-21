"""
The compressor, decompressor, stepper and projector networks.

The test that earns its place here is the exportability guard. Learned motion matching is worth
having over a database search because a few small networks can run anywhere, and that stops being
true the moment something in the forward pass cannot be exported to ONNX. A ``torch.where`` added
in good faith would cost nothing measurable in training and quietly close off the Unity-native
inference path, so the constraint is asserted rather than left in the module docstring.

Torch is optional here, as it is in ``test_pfnn_model``: the suite's stated property is that it
needs neither Unity nor venv extras, and learned motion matching must not be what takes that away.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from torch import nn

    import lmm_model
    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only on a bare interpreter
    HAS_TORCH = False

FEATURE_SIZE = 15
LATENT_SIZE = 8
POSE_SIZE = 21
CHARACTER_SIZE = 18


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class ExportabilityTests(unittest.TestCase):
    """Every leaf module must be one ONNX has always had -- see this module's docstring."""

    ALLOWED = ('Linear', 'ELU', 'ReLU')

    def leaf_modules(self, model):
        return [module for module in model.modules() if not list(module.children())]

    def assert_exportable(self, model):
        for module in self.leaf_modules(model):
            self.assertIn(type(module).__name__, self.ALLOWED,
                          f'{type(module).__name__} is not exportable to ONNX; see the module '
                          'docstring for why the forward pass is restricted')

    def test_the_compressor_is_linear_and_activations_only(self):
        self.assert_exportable(lmm_model.Compressor(POSE_SIZE, CHARACTER_SIZE, LATENT_SIZE))

    def test_the_decompressor_is_linear_and_activations_only(self):
        self.assert_exportable(lmm_model.Decompressor(FEATURE_SIZE, LATENT_SIZE, POSE_SIZE))

    def test_the_stepper_is_linear_and_activations_only(self):
        self.assert_exportable(lmm_model.Stepper(FEATURE_SIZE, LATENT_SIZE))

    def test_the_projector_is_linear_and_activations_only(self):
        self.assert_exportable(lmm_model.Projector(FEATURE_SIZE, LATENT_SIZE))

    def test_the_guard_would_catch_an_unexportable_layer(self):
        model = lmm_model.Mlp(4, 4, 8, 1)
        model.layers = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))

        with self.assertRaises(AssertionError):
            self.assert_exportable(model)


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class ShapeTests(unittest.TestCase):
    def test_the_compressor_takes_one_pose_in_two_spaces_and_returns_one_latent(self):
        compressor = lmm_model.Compressor(POSE_SIZE, CHARACTER_SIZE, LATENT_SIZE)

        self.assertEqual(compressor.input_size, POSE_SIZE + CHARACTER_SIZE)
        self.assertEqual(
            compressor.encode(torch.zeros(5, POSE_SIZE),
                              torch.zeros(5, CHARACTER_SIZE)).shape, (5, LATENT_SIZE))

    def test_the_decompressor_takes_a_query_and_a_latent_and_returns_a_pose(self):
        decompressor = lmm_model.Decompressor(FEATURE_SIZE, LATENT_SIZE, POSE_SIZE)

        self.assertEqual(decompressor.input_size, FEATURE_SIZE + LATENT_SIZE)
        self.assertEqual(
            decompressor.decode(torch.zeros(5, FEATURE_SIZE),
                                torch.zeros(5, LATENT_SIZE)).shape, (5, POSE_SIZE))

    def test_the_stepper_maps_the_state_onto_a_rate_of_the_same_width(self):
        stepper = lmm_model.Stepper(FEATURE_SIZE, LATENT_SIZE)

        self.assertEqual(stepper.input_size, FEATURE_SIZE + LATENT_SIZE)
        self.assertEqual(stepper.output_size, FEATURE_SIZE + LATENT_SIZE)
        self.assertEqual(
            stepper.rate(torch.zeros(5, FEATURE_SIZE),
                         torch.zeros(5, LATENT_SIZE)).shape, (5, FEATURE_SIZE + LATENT_SIZE))

    def test_the_projector_takes_a_query_and_returns_a_whole_state(self):
        projector = lmm_model.Projector(FEATURE_SIZE, LATENT_SIZE)

        self.assertEqual(projector.input_size, FEATURE_SIZE)
        self.assertEqual(projector.output_size, FEATURE_SIZE + LATENT_SIZE)

        features, latent = projector.project(torch.zeros(5, FEATURE_SIZE))
        self.assertEqual(features.shape, (5, FEATURE_SIZE))
        self.assertEqual(latent.shape, (5, LATENT_SIZE))

    def test_the_projector_answers_with_what_the_stepper_and_decompressor_take(self):
        # The stage takes the projector's answer and hands it straight to the other two, so a
        # width disagreeing here would be a runtime error a long way from its cause.
        projector = lmm_model.Projector(FEATURE_SIZE, LATENT_SIZE)
        stepper = lmm_model.Stepper(FEATURE_SIZE, LATENT_SIZE)

        self.assertEqual(projector.output_size, stepper.input_size)

    def test_the_stepper_takes_the_same_vector_the_decompressor_takes(self):
        # Not a coincidence worth preserving by luck: the stage carries one copy of (X, Z) and
        # hands it to both, so a second normalisation could not be got right in only one of them.
        stepper = lmm_model.Stepper(FEATURE_SIZE, LATENT_SIZE)
        decompressor = lmm_model.Decompressor(FEATURE_SIZE, LATENT_SIZE, POSE_SIZE)

        self.assertEqual(stepper.input_size, decompressor.input_size)

    def test_the_papers_activations_are_what_the_networks_are_built_with(self):
        compressor = lmm_model.Compressor(POSE_SIZE, CHARACTER_SIZE, LATENT_SIZE)
        decompressor = lmm_model.Decompressor(FEATURE_SIZE, LATENT_SIZE, POSE_SIZE)

        self.assertTrue(any(isinstance(m, nn.ELU) for m in compressor.layers))
        self.assertTrue(any(isinstance(m, nn.ReLU) for m in decompressor.layers))

    def test_the_reference_depths_are_what_the_networks_are_built_at(self):
        # Read off the reference implementation's shipped graphs: the decompressor is shallow and
        # wide, which looks backwards beside the deep projector and is not -- see lmm_model.
        self.assertEqual(
            len(lmm_model.Compressor(POSE_SIZE, CHARACTER_SIZE, LATENT_SIZE).linear_layers),
            lmm_model.COMPRESSOR_HIDDEN_LAYERS + 1)
        self.assertEqual(
            len(lmm_model.Decompressor(FEATURE_SIZE, LATENT_SIZE, POSE_SIZE).linear_layers),
            lmm_model.DECOMPRESSOR_HIDDEN_LAYERS + 1)
        self.assertEqual(
            len(lmm_model.Stepper(FEATURE_SIZE, LATENT_SIZE).linear_layers),
            lmm_model.STEPPER_HIDDEN_LAYERS + 1)
        self.assertEqual(
            len(lmm_model.Projector(FEATURE_SIZE, LATENT_SIZE).linear_layers),
            lmm_model.PROJECTOR_HIDDEN_LAYERS + 1)

    def test_the_projector_is_the_deepest_of_the_four(self):
        # It approximates a nearest-neighbour lookup, which is piecewise constant over as many
        # pieces as the database has frames, and depth is what buys the pieces.
        self.assertGreater(lmm_model.PROJECTOR_HIDDEN_LAYERS,
                           lmm_model.DECOMPRESSOR_HIDDEN_LAYERS)
        self.assertGreater(lmm_model.PROJECTOR_HIDDEN_LAYERS, lmm_model.STEPPER_HIDDEN_LAYERS)

    def test_no_hidden_layer_leaves_a_single_linear_map(self):
        model = lmm_model.Mlp(4, 3, 16, 0)

        self.assertEqual(len(model.linear_layers), 1)
        self.assertEqual(model(torch.zeros(2, 4)).shape, (2, 3))


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class ParameterTests(unittest.TestCase):
    def test_saved_parameters_reload_into_an_identical_model(self):
        torch.manual_seed(1)
        source = lmm_model.Decompressor(FEATURE_SIZE, LATENT_SIZE, POSE_SIZE)
        target = lmm_model.Decompressor(FEATURE_SIZE, LATENT_SIZE, POSE_SIZE)

        features, latent = torch.randn(4, FEATURE_SIZE), torch.randn(4, LATENT_SIZE)
        target.load_parameters(source.weights(), source.biases())

        with torch.no_grad():
            np.testing.assert_allclose(source.decode(features, latent).numpy(),
                                       target.decode(features, latent).numpy(),
                                       rtol=0, atol=1e-6)

    def test_one_array_per_linear_layer_is_stored(self):
        model = lmm_model.Mlp(4, 3, 16, 2)

        self.assertEqual(len(model.weights()), 3)
        self.assertEqual(len(model.biases()), 3)
        self.assertEqual(model.weights()[0].shape, (16, 4))
        self.assertEqual(model.weights()[-1].shape, (3, 16))

    def test_a_checkpoint_with_the_wrong_layer_count_is_refused(self):
        model = lmm_model.Mlp(4, 3, 16, 2)
        source = lmm_model.Mlp(4, 3, 16, 1)

        with self.assertRaises(ValueError) as raised:
            model.load_parameters(source.weights(), source.biases())
        self.assertIn('retrained', str(raised.exception))

    def test_a_checkpoint_with_the_wrong_width_is_refused_by_layer(self):
        model = lmm_model.Mlp(4, 3, 16, 1)
        source = lmm_model.Mlp(4, 3, 32, 1)

        with self.assertRaises(ValueError) as raised:
            model.load_parameters(source.weights(), source.biases())
        self.assertIn('layer 0', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
