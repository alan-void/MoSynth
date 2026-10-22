"""The motion field's requested-speed task dimension: the reward term, the fit, and the file format."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

# Attaching the PyCharm debugger costs a socket timeout on import of motion_field.action_predictor.
os.environ.setdefault('MOSYNTH_PYCHARM_DEBUG', '0')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_field import io as mfio  # noqa: E402


def _packed_velocity(x, y, z):
    """A packed per-second velocity whose character-frame slot moves at (x, y, z)."""
    v = np.zeros((4, 4), dtype=np.float32)
    v[0, :3] = (x, y, z)
    return v


class SpeedRewardTermTests(unittest.TestCase):
    def test_ground_speed_ignores_vertical_motion(self):
        from motion_field.field import ground_speed
        v = np.stack([_packed_velocity(3.0, 9.0, 4.0), _packed_velocity(0.0, -2.0, 0.0)])
        np.testing.assert_allclose(ground_speed(v), [5.0, 0.0], atol=1e-6)

    def test_penalty_is_zero_at_the_request_and_linear_either_side(self):
        from motion_field.field import speed_penalty
        penalty = speed_penalty(np.array([0.5, 1.0, 1.5, 3.0]), 1.0, speed_factor=2.0)
        np.testing.assert_allclose(penalty, [1.0, 0.0, 1.0, 4.0], atol=1e-6)

    def test_a_zero_factor_switches_the_term_off(self):
        from motion_field.field import speed_penalty
        np.testing.assert_array_equal(speed_penalty(np.array([0.0, 2.0]), 1.0, 0.0), [0.0, 0.0])

    def test_a_request_to_stand_pays_no_locomotion_bonus(self):
        from motion_field.field import request_gate
        np.testing.assert_allclose(request_gate(np.array([0.0, 0.25, 0.5, 2.0]), 0.5),
                                   [0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_a_request_is_bracketed_on_the_grid_and_clamped_past_its_ends(self):
        from motion_field.field import speed_bracket
        grid = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        self.assertEqual(speed_bracket(grid, 1.25), (1, 2, 0.25))
        self.assertEqual(speed_bracket(grid, -1.0), (0, 1, 0.0))
        lower, upper, alpha = speed_bracket(grid, 5.0)
        self.assertEqual((lower, upper), (1, 2))
        self.assertAlmostEqual(alpha, 1.0)
        self.assertEqual(speed_bracket(np.array([1.0], dtype=np.float32), 3.0), (0, 0, 0.0))


class SpeedValueFunctionFitTests(unittest.TestCase):
    """
    Three states, standing, walking at 1 m/s and running at 2 m/s. From anywhere,
    action `a` lands exactly on state `a` without turning, so the only thing that
    separates the actions is the speed they arrive at.
    """

    STATE_SPEEDS = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    THRESHOLD = 0.5
    GAMMA = 0.9

    def setUp(self):
        from motion_field.field import state_locomotion_scores
        from motion_field.trainer import speed_grid, train_value_function

        # build_tables always gives as many successor neighbours as actions, and the backup
        # relies on it; every neighbour of action a is state a.
        states = actions = neighbours = 3
        self.delta_yaw = np.zeros((states, actions), dtype=np.float32)
        self.indices = np.tile(np.arange(actions, dtype=np.int32)[np.newaxis, :, np.newaxis],
                               (states, 1, neighbours))
        self.weights = np.full((states, actions, neighbours), 1.0 / neighbours, dtype=np.float32)
        self.arrival_speed = np.tile(self.STATE_SPEEDS, (states, 1))
        velocities = np.stack([_packed_velocity(0.0, 0.0, speed) for speed in self.STATE_SPEEDS])
        self.state_scores = state_locomotion_scores(velocities, self.THRESHOLD)

        self.speeds = speed_grid(0.0, 2.0, 3)
        self.theta_count = 4  # includes a heading of exactly 0, at index 2
        self.values, self.scores = train_value_function(
            self.delta_yaw, self.indices, self.weights,
            state_scores=self.state_scores, locomotion_factor=1.0,
            arrival_speed=self.arrival_speed, speed_factor=1.0, speeds=self.speeds,
            locomotion_speed_threshold=self.THRESHOLD,
            theta_count=self.theta_count, epochs=400, gamma=self.GAMMA, device='cpu',
            residual_tolerance=1e-7)

    def _q_values(self, slot):
        """Q(s, a) at a goal straight ahead, for the request at `slot`."""
        from motion_field.field import request_gate, speed_penalty
        gate = request_gate(self.speeds[slot], self.THRESHOLD)
        rewards = (-speed_penalty(self.arrival_speed, self.speeds[slot], 1.0)
                   + gate * self.state_scores[np.newaxis, :])
        straight_ahead = 2
        return rewards + self.GAMMA * self.values[np.arange(3), straight_ahead, slot][np.newaxis, :]

    def test_the_value_function_has_a_slice_per_requested_speed(self):
        self.assertEqual(self.values.shape, (3, self.theta_count, 3))
        self.assertLess(self.scores[-1, 2], 1e-6)

    def test_the_policy_arrives_at_the_requested_speed(self):
        for slot, requested in enumerate(self.speeds):
            best = np.argmax(self._q_values(slot), axis=1)
            np.testing.assert_array_equal(
                self.STATE_SPEEDS[best], np.full(3, requested),
                err_msg=f'requested {requested} m/s')

    def test_runtime_interpolation_reads_the_requested_slice(self):
        from motion_field.field import MotionField
        vf = mfio.ValueFunctionData(values=self.values, scores=self.scores,
                                    theta_count=self.theta_count, gamma=self.GAMMA,
                                    k_neighbors=1, speeds=self.speeds)
        field = SimpleNamespace(value_function=vf)
        indices = np.array([[0], [1], [2]])
        weights = np.ones((3, 1), dtype=np.float32)
        theta = np.zeros(3, dtype=np.float32)

        exact = MotionField.interpolate_value(field, indices, weights, theta, 1.0)
        np.testing.assert_allclose(exact, self.values[:, 2, 1], atol=1e-5)

        halfway = MotionField.interpolate_value(field, indices, weights, theta, 1.5)
        np.testing.assert_allclose(halfway, 0.5 * (self.values[:, 2, 1] + self.values[:, 2, 2]),
                                   atol=1e-5)


class ValueFunctionFormatTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.messages = []

    def tearDown(self):
        self.folder.cleanup()

    def _path(self, name):
        return os.path.join(self.folder.name, name)

    def test_a_value_function_round_trips_with_its_speed_grid(self):
        values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        speeds = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
        mfio.save_value_function(self._path('new.mffield.npz'), values, np.zeros((1, 3)),
                                 theta_count=3, gamma=0.9, k_neighbors=5, speeds=speeds)

        loaded = mfio.load_value_function(self._path('new.mffield.npz'), log=self.messages.append)

        np.testing.assert_array_equal(loaded.values, values)
        np.testing.assert_array_equal(loaded.speeds, speeds)
        self.assertEqual(loaded.speed_count, 4)
        self.assertEqual(self.messages, [])

    def test_a_heading_only_value_function_is_refused_with_a_request_to_retrain(self):
        path = self._path('old.mffield.npz')
        np.savez(path, value_function=np.zeros((10, 17), dtype=np.float32),
                 scores=np.zeros((1, 3), dtype=np.float32), theta_count=np.int64(17),
                 gamma=np.float32(0.99), k_neighbors=np.int64(15))

        self.assertIsNone(mfio.load_value_function(path, log=self.messages.append))
        self.assertEqual(len(self.messages), 1)
        self.assertIn('heading-only', self.messages[0])
        self.assertIn('Retrain', self.messages[0])

    def test_values_that_disagree_with_their_grids_are_refused(self):
        path = self._path('mismatch.mffield.npz')
        np.savez(path, value_function=np.zeros((10, 17, 3), dtype=np.float32),
                 scores=np.zeros((1, 3), dtype=np.float32), theta_count=np.int64(17),
                 gamma=np.float32(0.99), k_neighbors=np.int64(15),
                 speeds=np.zeros(5, dtype=np.float32))

        self.assertIsNone(mfio.load_value_function(path, log=self.messages.append))
        self.assertIn('Retrain', self.messages[0])

    def test_saving_values_without_a_speed_axis_raises(self):
        with self.assertRaises(ValueError):
            mfio.save_value_function(self._path('bad.mffield.npz'), np.zeros((2, 3), np.float32),
                                     np.zeros((1, 3)), theta_count=3, gamma=0.9, k_neighbors=5,
                                     speeds=np.array([1.0], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
