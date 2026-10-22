"""Seeding a training run: every generator it can draw from, and the motion field trainer end to end."""

import os
import random
import sys
import tempfile
import unittest

import numpy as np

# Attaching the PyCharm debugger costs a socket timeout on import of motion_field.action_predictor.
os.environ.setdefault('MOSYNTH_PYCHARM_DEBUG', '0')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.seeding import seed_everything  # noqa: E402
from test_pose_set_importer import write_mmpose  # noqa: E402


def _draws():
    import torch
    return random.random(), float(np.random.rand()), float(torch.rand(1).item())


class SeedEverythingTests(unittest.TestCase):
    def test_the_same_seed_repeats_every_generator(self):
        seed_everything(7)
        first = _draws()
        seed_everything(7)
        self.assertEqual(_draws(), first)

    def test_a_different_seed_moves_every_generator(self):
        seed_everything(7)
        first = _draws()
        seed_everything(8)
        second = _draws()
        for a, b in zip(first, second):
            self.assertNotEqual(a, b)


class MotionFieldTrainerSeedTests(unittest.TestCase):
    """A tiny synthetic database, trained on the CPU in well under a second."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        poses = 24
        write_mmpose(os.path.join(self.folder.name, 'tiny.mmpose'), poses,
                     ['root', 'spine', 'foot_l', 'foot_r'], [-1, 0, 0, 0],
                     np.linspace(0.0, 1.0, poses), np.ones(poses))

    def tearDown(self):
        self.folder.cleanup()

    def _train(self, out_name, seed):
        from motion_field import trainer
        out_path = os.path.join(self.folder.name, out_name)
        summary = trainer.train(self.folder.name, 'tiny', out_path, k_neighbors=3, theta_count=5,
                                epochs=5, device='cpu', seed=seed)
        with np.load(out_path) as data:
            return summary, data['value_function'].copy(), data['scores'].copy()

    def test_the_same_seed_writes_the_same_value_function(self):
        summary_a, values_a, scores_a = self._train('a.mffield.npz', seed=3)
        _, values_b, scores_b = self._train('b.mffield.npz', seed=3)
        np.testing.assert_array_equal(values_a, values_b)
        np.testing.assert_array_equal(scores_a, scores_b)
        self.assertEqual(summary_a['seed'], 3)


if __name__ == '__main__':
    unittest.main()
