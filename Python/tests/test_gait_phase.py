"""Gait phase reconstruction from foot contacts."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gait_phase  # noqa: E402
from gait_phase import LEFT, RIGHT, TAU  # noqa: E402

FRAME_TIME = 1.0 / 30.0


def contacts_from_footfalls(n_frames: int, left_frames, right_frames, stance: int = 4):
    """
    Contact flags that go down on the given frames and stay down for ``stance`` frames.

    Only the rising edges matter to the phase, but building the flags this way keeps the
    fixtures readable as gaits rather than as bit patterns.
    """
    contacts = np.zeros((n_frames, 2), dtype=bool)
    for column, frames in ((LEFT, left_frames), (RIGHT, right_frames)):
        for frame in frames:
            contacts[frame:frame + stance, column] = True
    return contacts


class FootfallDetectionTests(unittest.TestCase):
    def test_a_rising_edge_is_a_footfall(self):
        flags = np.array([0, 0, 1, 1, 0, 1], dtype=bool)

        np.testing.assert_array_equal(gait_phase.footfall_frames(flags), [2, 5])

    def test_a_clip_that_starts_in_stance_has_no_footfall_there(self):
        flags = np.array([1, 1, 0, 1], dtype=bool)

        np.testing.assert_array_equal(gait_phase.footfall_frames(flags), [3])


class ClipPhaseTests(unittest.TestCase):
    def test_footfalls_land_on_the_half_cycle_marks(self):
        # Right at 10 and 30, left at 20: one full stride, right heel to right heel.
        contacts = contacts_from_footfalls(45, left_frames=[20], right_frames=[10, 30])

        phase = gait_phase.unwrapped_clip_phase(contacts)

        self.assertAlmostEqual(phase[10], 0.0, places=5)
        self.assertAlmostEqual(phase[20], np.pi, places=5)
        self.assertAlmostEqual(phase[30], TAU, places=5)

    def test_a_clip_opening_on_a_left_footfall_is_offset_by_half_a_cycle(self):
        contacts = contacts_from_footfalls(45, left_frames=[10], right_frames=[20])

        phase = gait_phase.unwrapped_clip_phase(contacts)

        self.assertAlmostEqual(phase[10], np.pi, places=5)
        self.assertAlmostEqual(phase[20], TAU, places=5)

    def test_phase_is_linear_between_footfalls(self):
        contacts = contacts_from_footfalls(45, left_frames=[20], right_frames=[10, 30])

        phase = gait_phase.unwrapped_clip_phase(contacts)

        self.assertAlmostEqual(phase[15], 0.5 * np.pi, places=5)

    def test_phase_keeps_advancing_past_the_last_footfall(self):
        contacts = contacts_from_footfalls(45, left_frames=[20], right_frames=[10, 30])

        phase = gait_phase.unwrapped_clip_phase(contacts)

        self.assertGreater(phase[40], phase[30], "a flat tail would read as the character stopping")
        self.assertLess(phase[0], phase[10], "and a flat head as it starting from nothing")

    def test_two_footfalls_of_the_same_foot_advance_a_whole_cycle(self):
        # The left foot never comes down, so the right heel to right heel gap is one stride.
        contacts = contacts_from_footfalls(45, left_frames=[], right_frames=[10, 30])

        phase = gait_phase.unwrapped_clip_phase(contacts)

        self.assertAlmostEqual(phase[30] - phase[10], TAU, places=5)

    def test_a_clip_with_no_measurable_cycle_reports_none(self):
        standing = np.ones((30, 2), dtype=bool)

        phase, rate = gait_phase.clip_phase(standing, FRAME_TIME)

        np.testing.assert_allclose(phase, 0.0)
        np.testing.assert_allclose(rate, 0.0)

    def test_wrapped_phase_stays_inside_one_cycle(self):
        contacts = contacts_from_footfalls(120, left_frames=[20, 60, 100], right_frames=[10, 40, 80])

        phase, _ = gait_phase.clip_phase(contacts, FRAME_TIME)

        self.assertTrue(np.all(phase >= 0.0))
        self.assertTrue(np.all(phase < TAU))

    def test_the_rate_does_not_spike_at_the_wrap(self):
        contacts = contacts_from_footfalls(120, left_frames=[20, 60, 100], right_frames=[10, 40, 80])

        phase, rate = gait_phase.clip_phase(contacts, FRAME_TIME)

        wraps = np.flatnonzero(np.diff(phase) < 0)
        self.assertGreater(wraps.size, 0, "the fixture should cover more than one cycle")
        self.assertTrue(np.all(rate > 0.0))
        self.assertLess(rate.max(), 4.0 * rate.mean())


class PoseSetPhaseTests(unittest.TestCase):
    def test_phase_never_carries_across_a_clip_boundary(self):
        first = contacts_from_footfalls(50, left_frames=[20], right_frames=[10, 30])
        second = contacts_from_footfalls(50, left_frames=[25], right_frames=[15, 35])
        contacts = np.concatenate([first, second])

        phase, _ = gait_phase.pose_set_phase(contacts, [(0, 50), (50, 100)], FRAME_TIME)

        # Clip 1 restarts its own cycle, so its right footfall sits on 0 as clip 0's did.
        self.assertAlmostEqual(phase[10], 0.0, places=5)
        self.assertAlmostEqual(phase[50 + 15], 0.0, places=5)

    def test_a_standing_clip_beside_a_walking_one_keeps_its_own_zero_rate(self):
        walking = contacts_from_footfalls(50, left_frames=[20], right_frames=[10, 30])
        standing = np.ones((20, 2), dtype=bool)
        contacts = np.concatenate([walking, standing])

        _, rate = gait_phase.pose_set_phase(contacts, [(0, 50), (50, 70)], FRAME_TIME)

        self.assertTrue(np.all(rate[:50] > 0.0))
        np.testing.assert_allclose(rate[50:], 0.0)


if __name__ == '__main__':
    unittest.main()
