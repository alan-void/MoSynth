"""The motion field's goal-heading convention, and how the greedy policy turns toward the goal."""

import os
import sys
import unittest

import numpy as np

# Attaching the PyCharm debugger costs a socket timeout on import of motion_field.action_predictor.
os.environ.setdefault('MOSYNTH_PYCHARM_DEBUG', '0')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_field.field import MotionField, root_yaw, travel_direction  # noqa: E402

FRAME_TIME = 1.0 / 30.0
BONES = 2  # packed rows: frame position, root position, frame rotation, root rotation


def _candidate(turn_degrees_per_frame, travel_degrees, speed=1.0):
    """
    One outcome of an action: the packed pose after one frame and its per-second velocity.

    :param turn_degrees_per_frame: how far the character frame turned this frame, positive to the
        right (Unity's SignedAngle about +Y)
    :param travel_degrees: travel direction measured from the new facing, positive to the right
    """
    x = np.zeros((BONES + 2, 4), dtype=np.float32)
    half = 0.5 * np.radians(turn_degrees_per_frame)
    x[2] = (0.0, np.sin(half), 0.0, np.cos(half))
    x[3] = (0.0, 0.0, 0.0, 1.0)

    v = np.zeros((BONES + 2, 4), dtype=np.float32)
    travel = np.radians(travel_degrees)
    v[0, :3] = (speed * np.sin(travel), 0.0, speed * np.cos(travel))
    return x, v


def _field(greedy_horizon):
    """A MotionField holding only the reward weights, which is all its scoring reads."""
    field = MotionField.__new__(MotionField)
    field.locomotion_factor = 1.0
    field.locomotion_speed_threshold = 0.5
    field.travel_factor = 0.5
    field.speed_factor = 1.0
    field.greedy_horizon = greedy_horizon
    return field


def _greedy_choice(field, theta, candidates):
    xs, vs = (np.stack(arrays) for arrays in zip(*candidates))
    arrival = np.ones(len(candidates), dtype=np.float32)
    return int(np.argmax(field.greedy_rewards(theta, 1.0, xs, vs, arrival, FRAME_TIME)))


class HeadingConventionTests(unittest.TestCase):
    """theta is -SignedAngle(facing, goal), so a goal on the right is a negative theta."""

    def test_a_turn_to_the_right_is_a_positive_frame_yaw(self):
        x, _ = _candidate(turn_degrees_per_frame=3.0, travel_degrees=0.0)
        self.assertAlmostEqual(np.degrees(root_yaw(x)), 3.0, places=4)

    def test_travel_to_the_right_of_facing_is_a_positive_beta(self):
        _, v = _candidate(turn_degrees_per_frame=0.0, travel_degrees=90.0)
        beta, _ = travel_direction(v, 0.5)
        self.assertAlmostEqual(np.degrees(float(beta)), 90.0, places=4)

    def test_walking_at_a_goal_on_the_right_pays_no_travel_penalty(self):
        field = _field(greedy_horizon=0.0)
        _, v = _candidate(turn_degrees_per_frame=0.0, travel_degrees=90.0)
        penalty = field.travel_penalty(np.array([-np.pi / 2]), v[np.newaxis])
        np.testing.assert_allclose(penalty, [0.0], atol=1e-5)


class GreedyTurningTests(unittest.TestCase):
    def test_a_goal_straight_ahead_is_walked_at_facing_it(self):
        candidates = [_candidate(0.0, 0.0), _candidate(0.0, 90.0), _candidate(0.0, -90.0),
                      _candidate(0.0, 180.0), _candidate(3.0, 0.0), _candidate(-3.0, 0.0)]
        self.assertEqual(_greedy_choice(_field(greedy_horizon=0.5), 0.0, candidates), 0)

    def test_a_goal_to_the_right_is_turned_toward_rather_than_strafed_toward(self):
        turn_right = _candidate(turn_degrees_per_frame=3.0, travel_degrees=0.0)
        strafe_right = _candidate(turn_degrees_per_frame=-0.5, travel_degrees=90.0)
        candidates = [turn_right, strafe_right]
        self.assertEqual(_greedy_choice(_field(greedy_horizon=0.5), -np.pi / 2, candidates), 0)

    def test_without_a_horizon_one_frame_of_turning_cannot_outweigh_the_travel_term(self):
        # Why the horizon exists: one frame turns a few degrees, so the heading term barely
        # separates the actions and the strafe, already travelling at the goal, wins.
        turn_right = _candidate(turn_degrees_per_frame=3.0, travel_degrees=0.0)
        strafe_right = _candidate(turn_degrees_per_frame=-0.5, travel_degrees=90.0)
        candidates = [turn_right, strafe_right]
        self.assertEqual(_greedy_choice(_field(greedy_horizon=0.0), -np.pi / 2, candidates), 1)

    def test_a_goal_on_the_left_is_turned_toward_leftward(self):
        candidates = [_candidate(3.0, 0.0), _candidate(-3.0, 0.0)]
        self.assertEqual(_greedy_choice(_field(greedy_horizon=0.5), np.pi / 2, candidates), 1)

    def test_aligned_with_the_goal_the_horizon_damps_turning(self):
        candidates = [_candidate(4.0, 0.0), _candidate(0.0, 0.0), _candidate(-4.0, 0.0)]
        self.assertEqual(_greedy_choice(_field(greedy_horizon=0.5), 0.0, candidates), 1)


if __name__ == '__main__':
    unittest.main()
