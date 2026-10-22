"""
The style network, its checkpoint, the trainer end to end, and the runtime step the C# stage calls.

The identity is the property everything else rests on: at zero network output and the neutral
style's identity posture, composing the deltas must hand back the input pose unchanged -- so a
character with the style turned off is exactly the character without the stage.

Torch is optional, as it is in ``test_lmm_model``.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_style_dataset import standard_set  # noqa: E402

try:
    import torch

    from lmm.fk import local_rotations
    from style import io as style_io
    from style import runtime
    from style import trainer
    from style.model import StyleNetwork
    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only on a bare interpreter
    HAS_TORCH = False

TINY = dict(hidden_units=16, layers=2, embed_dim=4, batch_size=64, device='cpu')


def train_tiny(out_path, epochs):
    training_set, labels = standard_set()
    summary = trainer.train_training_set(training_set, labels, out_path, epochs=epochs, **TINY)
    return training_set, summary


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class NetworkTests(unittest.TestCase):
    def test_an_untrained_network_outputs_exactly_zero(self):
        network = StyleNetwork(10, 7, n_styles=3, hidden_units=8, layers=2, embed_dim=4)
        out = network(torch.randn(5, 10), torch.tensor([0, 1, 2, 0, 1]))

        self.assertEqual(tuple(out.shape), (5, 7))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.folder.name, 'tiny.style.npz')
        cls.training_set, cls.summary = train_tiny(cls.path, epochs=3)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_the_summary_reports_what_was_trained(self):
        summary = self.summary
        for key in ('seconds', 'device', 'samples', 'styles_kept', 'styles_dropped'):
            self.assertIn(key, summary)
        self.assertEqual(summary['styles_kept'], ['style.Neutral', 'style.Sad'])
        self.assertIn('cadence', summary['styles_dropped']['style.Hop'])
        self.assertEqual(summary['epochs_run'], 3)
        self.assertTrue(np.isfinite(summary['train_loss']))
        self.assertTrue(np.isfinite(summary['val_loss']))
        self.assertEqual(summary['held_out_transition_clips'], 1)

    def test_the_checkpoint_round_trips(self):
        checkpoint = style_io.load_checkpoint(self.path)
        with tempfile.TemporaryDirectory() as folder:
            copy = os.path.join(folder, 'copy.style.npz')
            style_io.save_checkpoint(copy, checkpoint)
            again = style_io.load_checkpoint(copy)

        for name in ('style_names', 'gait_names', 'bone_names', 'styled_bone_names',
                     'contact_bone_names', 'hidden_units', 'layers', 'embed_dim', 'gate_k',
                     'frame_time', 'gait_sigma', 'gate_threshold'):
            self.assertEqual(getattr(again, name), getattr(checkpoint, name), name)
        for name in ('posture_rotations', 'speed_ratios', 'x_mean', 'y_std', 'parents',
                     'history_offsets', 'gate_reference', 'losses'):
            np.testing.assert_array_equal(getattr(again, name), getattr(checkpoint, name))
        self.assertEqual(sorted(again.parameters), sorted(checkpoint.parameters))
        self.assertEqual(checkpoint.contact_bone_names, ['lfoot', 'rfoot'])
        self.assertEqual(checkpoint.losses.shape, (3, 2))

    def test_a_missing_or_corrupt_checkpoint_loads_as_none(self):
        messages = []
        self.assertIsNone(style_io.load_checkpoint(os.path.join(self.folder.name, 'none.npz')))
        broken = os.path.join(self.folder.name, 'broken.style.npz')
        np.savez(broken, x_mean=np.zeros(3))

        self.assertIsNone(style_io.load_checkpoint(broken, log=messages.append))
        self.assertEqual(len(messages), 1)
        with self.assertRaises(FileNotFoundError):
            runtime.StylePolicy(broken, log=messages.append)

    def test_the_step_returns_the_contracted_shapes(self):
        policy = runtime.StylePolicy(self.path)
        n_bones, n_styled = len(policy.bone_names()), len(policy.styled_bone_names())
        d, h = policy.descriptor_size(), len(policy.history_offsets())
        frame = 60

        result = policy.step(self.training_set.rotations[frame].ravel().tolist(),
                             self.training_set.positions[frame].ravel().tolist(),
                             self.training_set.contacts[frame].tolist(),
                             self.training_set.root_velocity[frame].tolist(),
                             float(self.training_set.root_yaw_rate[frame]), 1.0, 6.0,
                             [0.0] * (h * d), [0, 1])

        self.assertEqual(len(result), 8)
        posture, dynamics, posture_height, dynamics_height, ratio, residual, gate, desc = result
        self.assertEqual((len(posture), len(dynamics)), (2 * n_styled * 4, 2 * n_styled * 4))
        self.assertEqual((len(posture_height), len(dynamics_height), len(ratio)), (2, 2, 2))
        self.assertEqual(len(residual), 8)
        self.assertIsInstance(gate, float)
        self.assertTrue(0.0 <= gate <= 1.0)
        self.assertEqual(len(desc), d)
        self.assertEqual(d, 3 + len(policy.contact_bone_names))
        self.assertEqual(policy.styled_bone_names()[0], policy.bone_names()[0])
        self.assertEqual(n_bones, 7)
        self.assertEqual(policy.history_offsets(), [0.1, 0.2])
        self.assertIsInstance(policy.describe(), str)

    def test_the_gate_is_open_on_training_inputs_and_shut_far_from_them(self):
        policy = runtime.StylePolicy(self.path)
        d = policy.descriptor_size()
        frame = 60
        args = (self.training_set.rotations[frame].ravel(),
                self.training_set.positions[frame].ravel(),
                self.training_set.contacts[frame], self.training_set.root_velocity[frame],
                float(self.training_set.root_yaw_rate[frame]))
        history = np.tile(policy.step(*args, 0.0, 6.0, np.zeros(2 * d), [0])[7], 2)

        self.assertGreater(policy.step(*args, 0.0, 2 * np.pi, history, [0])[6], 0.5)
        self.assertEqual(policy.step(*args, 0.0, 200.0, history, [0])[6], 0.0)

    def test_the_offline_check_runs(self):
        policy = runtime.StylePolicy(self.path)
        summary = runtime.check(policy, self.training_set, 'Sad', [0.0, 0.0, 1.0], frames=200,
                                log=lambda message: None)

        self.assertEqual(summary['frames'], 200)
        for key in ('posture_degrees', 'gate_mean', 'slide_before', 'slide_after_stylize'):
            self.assertTrue(np.isfinite(summary[key]), key)


@unittest.skipUnless(HAS_TORCH, 'torch is not installed')
class IdentityTests(unittest.TestCase):
    def test_zero_output_and_the_neutral_posture_leave_the_pose_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'untrained.style.npz')
            training_set, _ = train_tiny(path, epochs=0)
            policy = runtime.StylePolicy(path)

        neutral = policy.style_index('Neutral')
        n_styled = len(policy.styled_bone_names())
        d = policy.descriptor_size()
        spec = policy.spec

        for frame in (5, 70, 150, 250):
            result = policy.step(training_set.rotations[frame].ravel(),
                                 training_set.positions[frame].ravel(),
                                 training_set.contacts[frame], training_set.root_velocity[frame],
                                 float(training_set.root_yaw_rate[frame]), 0.5, 6.0,
                                 np.zeros(2 * d), [neutral])
            posture = np.asarray(result[0]).reshape(n_styled, 4)
            dynamics = np.asarray(result[1]).reshape(n_styled, 4)
            self.assertEqual(result[2], [0.0])
            self.assertEqual(result[3], [0.0])
            self.assertEqual(result[4], [1.0])
            self.assertEqual(result[5], [0.0] * 4)

            local = local_rotations(training_set.rotations[frame:frame + 1, spec.styled_bones],
                                    spec.styled_parents)[0]
            composed = runtime.compose(local, posture, dynamics)
            difference = Rotation.from_quat(composed) * Rotation.from_quat(local).inv()
            np.testing.assert_allclose(difference.magnitude(), 0.0, atol=1e-6)

    def test_compose_scales_each_delta_by_its_weight(self):
        delta = Rotation.from_rotvec([0.4, 0.0, 0.0]).as_quat()
        identity = np.array([0.0, 0.0, 0.0, 1.0])

        half = runtime.compose(identity, delta, identity, posture_weight=0.5)
        off = runtime.compose(identity, delta, delta, posture_weight=0.0, dynamics_weight=0.0)

        self.assertAlmostEqual(Rotation.from_quat(half).magnitude(), 0.2, places=6)
        self.assertAlmostEqual(Rotation.from_quat(off).magnitude(), 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
