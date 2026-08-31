"""
The phase function, the network that hangs off it, and the checkpoint they are written to.

The checkpoint tests need only numpy and run everywhere. The network tests need torch, which the
other suites in this folder deliberately do not, so they skip rather than fail when it is absent --
the stated property of this directory is that it needs no venv extras, and a PFNN should not be
what takes that away.
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pfnn_io  # noqa: E402

try:
    import torch  # noqa: E402

    import pfnn_model  # noqa: E402
    HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the machine, not on the code
    HAS_TORCH = False

TAU = 2.0 * math.pi


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class PhaseFunctionTests(unittest.TestCase):
    def test_the_weights_are_a_partition_of_one(self):
        # Anything else would make the blended network's scale depend on the phase.
        weights = pfnn_model.catmull_rom_weights(torch.linspace(0.0, TAU, 64))
        np.testing.assert_allclose(weights.sum(dim=1).numpy(), 1.0, atol=1e-6)

    def test_the_function_is_cyclic_in_the_phase(self):
        # The whole point of the four control points: the network at the end of a gait cycle is the
        # same object as at the start, not merely close to it.
        at_zero = pfnn_model.catmull_rom_weights(torch.tensor([0.0, 1.0, 2.5]))
        wrapped = pfnn_model.catmull_rom_weights(torch.tensor([TAU, TAU + 1.0, TAU + 2.5]))
        np.testing.assert_allclose(at_zero.numpy(), wrapped.numpy(), atol=1e-6)

    def test_the_function_is_continuous_across_a_control_point(self):
        # A step here would show up as a visible pop once per quarter cycle.
        seam = TAU / 4.0
        before = pfnn_model.catmull_rom_weights(torch.tensor([seam - 1e-5]))
        after = pfnn_model.catmull_rom_weights(torch.tensor([seam + 1e-5]))
        np.testing.assert_allclose(before.numpy(), after.numpy(), atol=1e-4)


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class BlendEquivalenceTests(unittest.TestCase):
    def test_blending_the_outputs_matches_blending_the_weights(self):
        # The optimisation the model is built around. If these ever diverge, the fast path is wrong
        # and the only symptom would be a network that trains to a worse loss for no visible reason.
        torch.manual_seed(7)
        network = pfnn_model.PhaseFunctionedNetwork(12, 7, hidden_units=16, dropout=0.0).eval()

        x = torch.randn(6, 12)
        phase = torch.rand(6) * TAU
        fast = network(x, phase)

        phase_weights = pfnn_model.catmull_rom_weights(phase)
        slow = []
        for sample in range(x.shape[0]):
            activations = x[sample]
            for index, layer in enumerate(network.layers):
                weight, bias = layer.blended_weights(phase_weights[sample])
                activations = weight @ activations + bias
                if index < len(network.layers) - 1:
                    activations = torch.nn.functional.elu(activations)
            slow.append(activations)

        np.testing.assert_allclose(fast.detach().numpy(),
                                   torch.stack(slow).detach().numpy(), atol=1e-5)


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class FittingTests(unittest.TestCase):
    def test_the_network_can_overfit_a_handful_of_samples(self):
        # Not a quality check -- a sanity check that gradients reach every control point. A model
        # wired up wrongly still produces plausible per-frame losses on real data, because a pose
        # barely changes in a thirtieth of a second, but it cannot memorise 32 samples.
        torch.manual_seed(11)
        network = pfnn_model.PhaseFunctionedNetwork(12, 7, hidden_units=32, dropout=0.0)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2)

        x = torch.randn(32, 12)
        phase = torch.rand(32) * TAU
        y = torch.randn(32, 7)

        first = float(torch.nn.functional.mse_loss(network(x, phase), y))
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(network(x, phase), y)
            loss.backward()
            optimizer.step()

        self.assertLess(float(loss), first / 10.0)

    def test_every_control_point_receives_gradient(self):
        # Phases are drawn across the whole cycle, so a control point with no gradient means the
        # scatter in catmull_rom_weights is not reaching it.
        torch.manual_seed(13)
        network = pfnn_model.PhaseFunctionedNetwork(5, 3, hidden_units=8, dropout=0.0)
        phase = torch.linspace(0.0, TAU, 32)

        network(torch.randn(32, 5), phase).pow(2).mean().backward()

        for layer in network.layers:
            per_control_point = layer.weights.grad.abs().sum(dim=(1, 2))
            self.assertTrue(bool((per_control_point > 0).all()), 'a control point got no gradient')


class CheckpointTests(unittest.TestCase):
    """Numpy only, so these run wherever the rest of this folder does."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.shapes = [(4, 8, 5), (4, 8, 8), (4, 3, 8)]
        self.weights = [rng.standard_normal(shape).astype(np.float32) for shape in self.shapes]
        self.biases = [rng.standard_normal(shape[:2]).astype(np.float32) for shape in self.shapes]
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directory.cleanup()

    def write(self) -> str:
        path = os.path.join(self.directory.name, 'test.pfnn.npz')
        pfnn_io.save_checkpoint(
            path, self.weights, self.biases,
            x_mean=np.zeros(5, dtype=np.float32), x_std=np.ones(5, dtype=np.float32),
            y_mean=np.zeros(3, dtype=np.float32), y_std=np.ones(3, dtype=np.float32),
            bone_names=['root', 'spine', 'foot'],
            window_offsets=np.array([-10, 0, 10]), frame_time=1.0 / 30.0,
            hidden_units=8, dropout=0.3, losses=[(1.0, 2.0), (0.5, 0.8)])
        return path

    def test_a_checkpoint_reads_back_as_what_was_written(self):
        checkpoint = pfnn_io.load_checkpoint(self.write())

        self.assertIsNotNone(checkpoint)
        for saved, loaded in zip(self.weights, checkpoint.weights):
            np.testing.assert_array_equal(saved, loaded)
        for saved, loaded in zip(self.biases, checkpoint.biases):
            np.testing.assert_array_equal(saved, loaded)

        self.assertEqual(checkpoint.bone_names, ['root', 'spine', 'foot'])
        np.testing.assert_array_equal(checkpoint.window_offsets, [-10, 0, 10])
        self.assertEqual(checkpoint.hidden_units, 8)
        self.assertAlmostEqual(checkpoint.frame_time, 1.0 / 30.0, places=6)
        self.assertAlmostEqual(checkpoint.final_loss, 0.8, places=6)
        self.assertEqual((checkpoint.input_size, checkpoint.output_size, checkpoint.n_bones),
                         (5, 3, 3))

    def test_bone_names_survive_without_allow_pickle(self):
        # They are stored as a fixed-width unicode array on purpose: reading a checkpoint must never
        # mean executing what is inside it.
        with np.load(self.write(), allow_pickle=False) as data:
            self.assertEqual([str(name) for name in data['bone_names']],
                             ['root', 'spine', 'foot'])

    def test_a_missing_file_is_reported_rather_than_raised(self):
        # Callers run inside Py.GIL() from Unity, where an exception arrives as an opaque managed
        # error; a stage that declines to run with a legible message is the useful behaviour.
        messages = []
        self.assertIsNone(pfnn_io.load_checkpoint(
            os.path.join(self.directory.name, 'absent.pfnn.npz'), log=messages.append))

    def test_a_file_from_an_older_layout_is_reported_rather_than_raised(self):
        path = os.path.join(self.directory.name, 'stale.pfnn.npz')
        np.savez(path, something_else=np.zeros(3, dtype=np.float32))

        messages = []
        self.assertIsNone(pfnn_io.load_checkpoint(path, log=messages.append))
        self.assertTrue(any('could not read checkpoint' in message for message in messages))

    def test_the_wrong_number_of_layers_is_refused(self):
        with self.assertRaises(ValueError):
            pfnn_io.save_checkpoint(
                os.path.join(self.directory.name, 'short.pfnn.npz'),
                self.weights[:2], self.biases[:2],
                np.zeros(5), np.ones(5), np.zeros(3), np.ones(3),
                ['root'], np.array([0]), 1.0 / 30.0, 8, 0.3, [(1.0, 1.0)])


if __name__ == '__main__':
    unittest.main()
